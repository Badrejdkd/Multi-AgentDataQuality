import os
import sys
import glob
import json
import random
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import numpy as np
from django.utils.encoding import escape_uri_path
import io

# ── Ajouter cette fonction utilitaire en haut du fichier ──
def clean_for_json(obj):
    """Convertit les types numpy/pandas en types Python natifs."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    else:
        return obj

# ── Ajouter la racine du projet au path ──
BASE = settings.BASE_DIR.parent  # racine MULTIAGENT_DATA_PLATFORM
sys.path.insert(0, str(BASE))

# ── Importer les vrais agents ──
try:
    from agents.extraction_agent import ExtractionAgent
    from agents.cleaning_agent   import CleaningAgent
    from agents.quality_agent    import QualityAgent
    from agents.storage_agent    import StorageAgent
    AGENTS_OK = True
except ImportError as e:
    print(f"[WARN] Import agents échoué : {e}")
    AGENTS_OK = False

DATA_DIR = str(settings.BASE_DIR / 'data')
LOGS_DIR = str(settings.BASE_DIR / 'logs')

# Types de BDD supportés
DB_TYPES = ["sqlite", "mysql", "postgresql", "oracle", "sql_server"]


# ════════════════════════════════════════════
#  AGENTS FALLBACK (si import échoue)
# ════════════════════════════════════════════

class _CleaningAgent:
    def remove_duplicates(self, df):
        b = len(df); df = df.drop_duplicates(); return df, b - len(df)
    def fill_missing_values(self, df):
        n = 0
        for col in df.columns:
            m = int(df[col].isnull().sum())
            if m:
                n += m
                df[col] = df[col].fillna("unknown") if df[col].dtype == "object" else df[col].fillna(df[col].median())
        return df, n
    def normalize_text(self, df):
        cols = list(df.select_dtypes(include="object").columns)
        for c in cols:
            df[c] = df[c].str.strip().str.lower()
        return df, cols
    def clean_table(self, df):
        df, d = self.remove_duplicates(df)
        df, f = self.fill_missing_values(df)
        df, c = self.normalize_text(df)
        return df, {"duplicates_removed": d, "missing_filled": f, "columns_normalized": c}

class _QualityAgent:
    def detect_missing_values(self, df):
        return {k: int(v) for k, v in df.isnull().sum().items()}
    def detect_duplicates(self, df):
        return int(df.duplicated().sum())
    def detect_outliers(self, df):
        out = {}
        for col in df.select_dtypes(include=["int64", "float64"]).columns:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            out[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        return out
    def quality_score(self, df):
        total = df.shape[0] * df.shape[1]
        return round(100 - ((int(df.isnull().sum().sum()) + int(df.duplicated().sum())) / total * 100), 2)
    def quality_report(self, df):
        return {
            "rows":           len(df),
            "columns":        len(df.columns),
            "missing_values": self.detect_missing_values(df),
            "duplicates":     self.detect_duplicates(df),
            "outliers":       self.detect_outliers(df),
            "quality_score":  self.quality_score(df),
        }

class _StorageAgent:
    def save_to_csv(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        return path
    def save_raw_table(self, df, t):
        return self.save_to_csv(df, f"{DATA_DIR}/raw/{t}.csv")
    def save_cleaned_table(self, df, t):
        return self.save_to_csv(df, f"{DATA_DIR}/cleaned/{t}.csv")
    def save_quality_report(self, r, t):
        path = f"{DATA_DIR}/../reports/{t}_quality.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        return path

# Instances globales
_cleaner = CleaningAgent() if AGENTS_OK else _CleaningAgent()
_quality = QualityAgent()  if AGENTS_OK else _QualityAgent()
_storage = StorageAgent()  if AGENTS_OK else _StorageAgent()


# ════════════════════════════════════════════
#  HELPER — créer ExtractionAgent à la volée
# ════════════════════════════════════════════

def get_extractor(db_type, db_name):
    """
    Retourne un ExtractionAgent connecté.
    Si les agents ne sont pas importables, lève une exception claire.
    """
    if not AGENTS_OK:
        raise RuntimeError("ExtractionAgent non disponible (vérifier les imports)")
    return ExtractionAgent(db_type=db_type, db_name=db_name)


# ════════════════════════════════════════════
#  VUES — PAGES
# ════════════════════════════════════════════

def dashboard(request):
    raw_count     = len(glob.glob(f"{DATA_DIR}/raw/*.csv"))
    cleaned_count = len(glob.glob(f"{DATA_DIR}/cleaned/*.csv"))
    report_count  = len(glob.glob(f"{DATA_DIR}/../reports/*.json"))
    log_lines = []
    log_path = f"{LOGS_DIR}/pipeline.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_lines = f.readlines()[-10:]
    return render(request, "dashboard.html", {
        "page":          "dashboard",
        "db_types":      DB_TYPES,
        "raw_count":     raw_count,
        "cleaned_count": cleaned_count,
        "report_count":  report_count,
        "log_lines":     log_lines,
    })


def extraction(request):
    return render(request, "extraction.html", {
        "page":     "extraction",
        "db_types": DB_TYPES,
    })


def cleaning(request):
    # Tables depuis les fichiers raw déjà extraits
    raw_files = [
        os.path.basename(f).replace(".csv", "") 
        for f in glob.glob(f"{DATA_DIR}/raw/*.csv")
    ]
    
    # Tables depuis la BDD (si accessible)
    db_tables = []
    db_type = request.GET.get("db_type", "")
    db_name = request.GET.get("db_name", "")
    
    if db_type and db_name and AGENTS_OK:
        try:
            extractor = get_extractor(db_type, db_name)
            db_tables = extractor.get_all_tables()["TABLE_NAME"].tolist()
        except Exception:
            pass

    # Fusionner les deux sources sans doublons
    all_tables = list(set(raw_files + db_tables))

    return render(request, "cleaning.html", {
        "page":       "cleaning",
        "raw_files":  raw_files,    # tables déjà extraites (CSV dispo)
        "db_tables":  db_tables,    # tables dans la BDD
        "all_tables": all_tables,   # toutes les tables
        "db_types":   DB_TYPES,
    })


def quality(request):
    raw_files = [
        os.path.basename(f).replace(".csv", "")
        for f in glob.glob(f"{DATA_DIR}/raw/*.csv")
    ]
    
    cleaned_files = [
        os.path.basename(f).replace(".csv", "")
        for f in glob.glob(f"{DATA_DIR}/cleaned/*.csv")
    ]

    return render(request, "quality.html", {
        "page":          "quality",
        "raw_files":     raw_files,
        "cleaned_files": cleaned_files,
        "db_types":      DB_TYPES,
    })


def pipeline(request):
    return render(request, "pipeline.html", {
        "page":     "pipeline",
        "db_types": DB_TYPES,
    })


def logs_view(request):
    log_lines = []
    log_path = f"{LOGS_DIR}/pipeline.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_lines = f.readlines()
    return render(request, "logs.html", {
        "page":      "logs",
        "log_lines": log_lines,
    })


# ════════════════════════════════════════════
#  VUES — API
# ════════════════════════════════════════════

@csrf_exempt
def run_extraction(request):
    """
    POST params :
        db_type   : sqlite | mysql | postgresql | oracle | sql_server
        db_name   : nom/chemin de la BDD
        table_name: nom de la table (optionnel → liste toutes les tables)
        mode      : "single" | "all" | "llm"
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    db_type    = request.POST.get("db_type",    "sqlite")
    db_name    = request.POST.get("db_name",    "test.db.sqlite")
    table_name = request.POST.get("table_name", "")
    mode       = request.POST.get("mode",       "single")

    try:
        extractor = get_extractor(db_type, db_name)
    except Exception as e:
        return JsonResponse({"error": f"Connexion BDD échouée : {str(e)}"}, status=500)

    try:
        # ── Mode 1 : extraire une seule table ──
        if mode == "single" and table_name:
            df   = extractor.extract_table(table_name)
            path = _storage.save_raw_table(df, table_name)
            preview = df.head(10).where(df.head(10).notna(), None).to_dict(orient="records")
            return JsonResponse({
                "success":  True,
                "mode":     "single",
                "table":    table_name,
                "rows":     len(df),
                "columns":  list(df.columns),
                "saved_to": path,
                "preview":  preview,
            })

        # ── Mode 2 : extraire toutes les tables ──
        elif mode == "all":
            result = extractor.extract_all_data_to_csv(f"{DATA_DIR}/raw")
            tables = extractor.get_all_tables()["TABLE_NAME"].tolist()
            return JsonResponse({
                "success": True,
                "mode":    "all",
                "tables":  tables,
                "message": result,
            })

        # ── Mode 3 : LLM décide quelles tables extraire ──
        elif mode == "llm":
            result = extractor.extract_with_ollama_to_csv(f"{DATA_DIR}/raw")
            # S'assurer que result est un dictionnaire
            if isinstance(result, dict):
                return JsonResponse({
                    "success": True,
                    "mode":    "llm",
                    "message": result,  # C'est déjà un dict
                })
            else:
                return JsonResponse({
                    "success": True,
                    "mode":    "llm",
                    "message": {"result": str(result)},
                })

        # ── Mode 4 : lister les tables disponibles ──
        elif mode == "list":
            tables_df = extractor.get_all_tables()
            tables    = tables_df["TABLE_NAME"].tolist()
            return JsonResponse({
                "success": True,
                "mode":    "list",
                "tables":  tables,
            })

        else:
            return JsonResponse({"error": "Paramètres invalides (mode ou table_name manquant)"}, status=400)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def run_cleaning(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    table_name = request.POST.get("table_name", "")
    if not table_name:
        return JsonResponse({"error": "table_name requis"}, status=400)

    raw_path = f"{DATA_DIR}/raw/{table_name}.csv"
    if not os.path.exists(raw_path):
        return JsonResponse({"error": f"Fichier introuvable : {raw_path}"}, status=404)

    try:
        df               = pd.read_csv(raw_path)
        before           = len(df)
        df_clean, stats  = _cleaner.clean_table(df)
        path             = _storage.save_cleaned_table(df_clean, table_name)

        # Preview : remplacer NaN par None
        preview = df_clean.head(10).where(df_clean.head(10).notna(), None).to_dict(orient="records")

        # ── FIX : nettoyer avant JSON ──
        return JsonResponse(clean_for_json({
            "success":            True,
            "table":              table_name,
            "rows_before":        before,
            "rows_after":         len(df_clean),
            "duplicates_removed": stats["duplicates_removed"],
            "missing_filled":     stats["missing_filled"],
            "columns_normalized": stats["columns_normalized"],
            "saved_to":           path,
            "preview":            preview,
        }))
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def run_quality(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    table_name = request.POST.get("table_name", "")
    source     = request.POST.get("source", "raw")

    if not table_name:
        return JsonResponse({"error": "table_name requis"}, status=400)

    file_path = f"{DATA_DIR}/cleaned/{table_name}.csv" if source == "cleaned" else f"{DATA_DIR}/raw/{table_name}.csv"

    if not os.path.exists(file_path):
        return JsonResponse({"error": f"Fichier introuvable : {file_path}"}, status=404)

    try:
        df     = pd.read_csv(file_path)
        report = _quality.quality_report(df)
        _storage.save_quality_report(report, table_name)

        # ── FIX : nettoyer avant JSON ──
        return JsonResponse(clean_for_json({
            "success": True,
            "table":   table_name,
            "source":  source,
            "report":  report,
        }))
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def run_pipeline(request):
    """
    Pipeline complet :
    1. Connexion BDD via ExtractionAgent
    2. Liste toutes les tables
    3. QualityAgent → rapport
    4. CleaningAgent → nettoyage
    5. StorageAgent → sauvegarde
    6. Log
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    db_type = request.POST.get("db_type", "sqlite")
    db_name = request.POST.get("db_name", "test.db.sqlite")

    os.makedirs(LOGS_DIR, exist_ok=True)

    try:
        extractor  = get_extractor(db_type, db_name)
        tables_df  = extractor.get_all_tables()
        table_list = tables_df["TABLE_NAME"].tolist()
    except Exception as e:
        return JsonResponse({"error": f"Connexion BDD échouée : {str(e)}"}, status=500)

    results = []

    for table in table_list:
        try:
            # 1. Extraction
            df = extractor.extract_table(table)
            _storage.save_raw_table(df, table)

            # 2. Qualité
            report = _quality.quality_report(df)
            _storage.save_quality_report(report, table)

            # 3. Nettoyage
            df_clean, stats = _cleaner.clean_table(df)
            _storage.save_cleaned_table(df_clean, table)

            # 4. Log
            with open(f"{LOGS_DIR}/pipeline.log", "a") as log:
                log.write(
                    f"[PIPELINE] '{table}' — score:{report['quality_score']}% "
                    f"— dups:{stats['duplicates_removed']} "
                    f"— missing:{stats['missing_filled']}\n"
                )

            results.append({
                "table":              table,
                "rows":               report["rows"],
                "quality_score":      report["quality_score"],
                "duplicates_removed": stats["duplicates_removed"],
                "missing_filled":     stats["missing_filled"],
                "status":             "✅ Done",
            })

        except Exception as e:
            with open(f"{LOGS_DIR}/pipeline.log", "a") as log:
                log.write(f"[ERROR] '{table}' — {str(e)}\n")
            results.append({
                "table":  table,
                "status": f"❌ Erreur : {str(e)}",
                "rows": 0, 
                "quality_score": 0,
                "duplicates_removed": 0, 
                "missing_filled": 0,
            })

    return JsonResponse({"success": True, "results": results, "db": db_name})


# ════════════════════════════════════════════
#  FILTRAGE DYNAMIQUE DES DONNÉES
# ════════════════════════════════════════════

@csrf_exempt
def filter_data(request):
    """
    Filtre dynamique des données avec multiples critères
    POST params :
        table_name : nom de la table
        source : raw | cleaned
        filters : JSON des filtres (ex: {"age": ">25", "name": "contains:John"})
        columns : liste des colonnes à afficher (optionnel)
        limit : nombre max de lignes
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    table_name = request.POST.get("table_name", "")
    source = request.POST.get("source", "raw")
    filters_json = request.POST.get("filters", "{}")
    columns_json = request.POST.get("columns", "[]")
    limit = int(request.POST.get("limit", 1000))

    if not table_name:
        return JsonResponse({"error": "table_name requis"}, status=400)

    # Charger le fichier CSV
    file_path = f"{DATA_DIR}/{source}/{table_name}.csv"
    if not os.path.exists(file_path):
        return JsonResponse({"error": f"Fichier introuvable : {file_path}"}, status=404)

    try:
        df = pd.read_csv(file_path)
        filters = json.loads(filters_json)
        columns = json.loads(columns_json)

        # Appliquer les filtres
        filtered_df = apply_filters(df, filters)

        # Sélectionner les colonnes
        if columns and len(columns) > 0:
            # Vérifier que toutes les colonnes existent
            valid_columns = [c for c in columns if c in filtered_df.columns]
            if valid_columns:
                filtered_df = filtered_df[valid_columns]

        # Limiter le nombre de lignes
        filtered_df = filtered_df.head(limit)

        # Statistiques de filtrage
        stats = {
            "original_rows": len(df),
            "filtered_rows": len(filtered_df),
            "original_columns": len(df.columns),
            "filtered_columns": len(filtered_df.columns),
            "filters_applied": filters,
        }

        # Preview
        preview = filtered_df.head(20).where(filtered_df.head(20).notna(), None).to_dict(orient="records")

        return JsonResponse(clean_for_json({
            "success": True,
            "table": table_name,
            "source": source,
            "stats": stats,
            "preview": preview,
            "columns": list(filtered_df.columns),
        }))

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def apply_filters(df, filters):
    """Applique les filtres dynamiques au DataFrame"""
    filtered_df = df.copy()

    for column, condition in filters.items():
        if column not in df.columns:
            continue

        if isinstance(condition, dict):
            # Filtres multiples sur la même colonne
            for op, value in condition.items():
                filtered_df = apply_single_filter(filtered_df, column, op, value)
        else:
            # Filtre simple (égalité)
            filtered_df = apply_single_filter(filtered_df, column, "eq", condition)

    return filtered_df


def apply_single_filter(df, column, operator, value):
    """Applique un opérateur de filtrage spécifique"""
    try:
        if operator == "eq":  # Égal à
            return df[df[column] == value]
        elif operator == "neq":  # Différent de
            return df[df[column] != value]
        elif operator == "gt":  # Plus grand que
            return df[df[column] > float(value)]
        elif operator == "gte":  # Plus grand ou égal
            return df[df[column] >= float(value)]
        elif operator == "lt":  # Plus petit que
            return df[df[column] < float(value)]
        elif operator == "lte":  # Plus petit ou égal
            return df[df[column] <= float(value)]
        elif operator == "contains":  # Contient (texte)
            return df[df[column].astype(str).str.contains(value, case=False, na=False)]
        elif operator == "startswith":  # Commence par
            return df[df[column].astype(str).str.startswith(value, na=False)]
        elif operator == "endswith":  # Termine par
            return df[df[column].astype(str).str.endswith(value, na=False)]
        elif operator == "isnull":  # Est null
            return df[df[column].isna()]
        elif operator == "notnull":  # N'est pas null
            return df[df[column].notna()]
        elif operator == "between":  # Entre deux valeurs
            if isinstance(value, list) and len(value) == 2:
                return df[(df[column] >= float(value[0])) & (df[column] <= float(value[1]))]
        elif operator == "in":  # Dans une liste
            if isinstance(value, list):
                return df[df[column].isin(value)]
        elif operator == "notin":  # Pas dans une liste
            if isinstance(value, list):
                return df[~df[column].isin(value)]
    except:
        # En cas d'erreur, retourner le df inchangé
        pass
    
    return df


# ════════════════════════════════════════════
#  TÉLÉCHARGEMENT DES FICHIERS
# ════════════════════════════════════════════

@csrf_exempt
def download_csv(request):
    """
    Télécharger un fichier CSV avec filtres optionnels
    GET params :
        table_name : nom de la table
        source : raw | cleaned | filtered
        format : csv | excel | json
        filters : JSON des filtres (optionnel)
    """
    table_name = request.GET.get("table_name", "")
    source = request.GET.get("source", "raw")
    file_format = request.GET.get("format", "csv")
    filters_json = request.GET.get("filters", "{}")

    if not table_name:
        return JsonResponse({"error": "table_name requis"}, status=400)

    # Chemin du fichier source
    if source == "filtered":
        # Pour les données filtrées, on utilise le fichier raw ou cleaned comme base
        base_source = request.GET.get("base_source", "raw")
        file_path = f"{DATA_DIR}/{base_source}/{table_name}.csv"
    else:
        file_path = f"{DATA_DIR}/{source}/{table_name}.csv"

    if not os.path.exists(file_path):
        return JsonResponse({"error": f"Fichier introuvable : {file_path}"}, status=404)

    try:
        df = pd.read_csv(file_path)

        # Appliquer les filtres si fournis
        if filters_json and filters_json != "{}":
            filters = json.loads(filters_json)
            df = apply_filters(df, filters)
            filename = f"{table_name}_filtered"
        else:
            filename = f"{table_name}_{source}"

        # Générer le fichier selon le format demandé
        if file_format == "csv":
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="{escape_uri_path(filename)}.csv"'
            df.to_csv(response, index=False, encoding='utf-8-sig')

        elif file_format == "excel":
            response = HttpResponse(content_type='application/vnd.ms-excel')
            response['Content-Disposition'] = f'attachment; filename="{escape_uri_path(filename)}.xlsx"'
            with pd.ExcelWriter(response, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')

        elif file_format == "json":
            response = HttpResponse(content_type='application/json')
            response['Content-Disposition'] = f'attachment; filename="{escape_uri_path(filename)}.json"'
            json_data = df.to_json(orient='records', date_format='iso', indent=2)
            response.write(json_data)

        else:
            return JsonResponse({"error": "Format non supporté"}, status=400)

        # Ajouter des en-têtes pour éviter la mise en cache
        response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response['Pragma'] = 'no-cache'
        response['Expires'] = '0'

        return response

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def get_column_stats(request):
    """
    Obtenir les statistiques d'une colonne pour les filtres
    """
    table_name = request.GET.get("table_name", "")
    source = request.GET.get("source", "raw")
    column = request.GET.get("column", "")

    if not table_name or not column:
        return JsonResponse({"error": "table_name et column requis"}, status=400)

    file_path = f"{DATA_DIR}/{source}/{table_name}.csv"
    if not os.path.exists(file_path):
        return JsonResponse({"error": "Fichier introuvable"}, status=404)

    try:
        df = pd.read_csv(file_path)
        
        if column not in df.columns:
            return JsonResponse({"error": f"Colonne '{column}' introuvable"}, status=400)

        col_data = df[column]
        stats = {}

        if pd.api.types.is_numeric_dtype(col_data):
            # Statistiques pour colonnes numériques
            stats = {
                "type": "numeric",
                "min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                "max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                "mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                "median": float(col_data.median()) if not pd.isna(col_data.median()) else None,
                "unique_values": int(col_data.nunique()),
                "null_count": int(col_data.isna().sum()),
            }
        else:
            # Statistiques pour colonnes textuelles
            value_counts = col_data.value_counts().head(20).to_dict()
            stats = {
                "type": "text",
                "unique_values": int(col_data.nunique()),
                "null_count": int(col_data.isna().sum()),
                "top_values": {str(k): int(v) for k, v in value_counts.items() if pd.notna(k)},
            }

        return JsonResponse(clean_for_json({
            "success": True,
            "column": column,
            "stats": stats,
        }))

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
# ════════════════════════════════════════════
#  VUES LLM POUR EXTRACTION ET FILTRAGE INTELLIGENTS
# ════════════════════════════════════════════

@csrf_exempt
def llm_select_tables(request):
    """
    POST endpoint pour que le LLM choisisse les tables
    Body: {
        "db_type": "sqlite",
        "db_name": "test.db.sqlite",
        "prompt": "Je veux les tables clients et commandes",
        "extract": true/false  # Si true, extrait directement
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        # Essayer de parser le body JSON
        if request.body:
            try:
                data = json.loads(request.body)
            except:
                data = request.POST.dict()
        else:
            data = request.POST.dict()
    except:
        data = request.POST.dict()

    db_type = data.get("db_type", "sqlite")
    db_name = data.get("db_name", "test.db.sqlite")
    prompt = data.get("prompt", "")
    should_extract = data.get("extract", False)

    if not prompt:
        return JsonResponse({"error": "Prompt requis"}, status=400)

    try:
        extractor = get_extractor(db_type, db_name)
    except Exception as e:
        return JsonResponse({"error": f"Connexion BDD échouée : {str(e)}"}, status=500)

    try:
        # Vérifier si la méthode existe dans l'extracteur
        if should_extract and hasattr(extractor, 'extract_with_llm_selection'):
            # Extraire avec la sélection LLM
            result = extractor.extract_with_llm_selection(f"{DATA_DIR}/raw", prompt)
        elif hasattr(extractor, 'select_tables_with_llm'):
            # Juste sélectionner
            result = extractor.select_tables_with_llm(prompt)
        else:
            # Fallback: méthode simple
            tables_df = extractor.get_all_tables()
            table_list = tables_df['TABLE_NAME'].tolist()
            
            # Simulation de sélection (prendre les 3 premières tables)
            selected = table_list[:min(3, len(table_list))]
            
            result = {
                "tables_in_db": table_list,
                "selected_tables": selected,
                "llm_raw_response": "Méthode LLM non disponible - sélection par défaut",
                "prompt_used": prompt
            }
            
            if should_extract:
                os.makedirs(f"{DATA_DIR}/raw", exist_ok=True)
                extracted = []
                for table in selected:
                    try:
                        df = extractor.extract_table(table)
                        df.to_csv(f"{DATA_DIR}/raw/{table}.csv", index=False)
                        extracted.append(table)
                    except:
                        pass
                result["extracted_tables"] = extracted

        return JsonResponse(clean_for_json({
            "success": True,
            "result": result
        }))

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def llm_generate_filters(request):
    """
    POST endpoint pour que le LLM génère des filtres
    Body: {
        "db_type": "sqlite",
        "db_name": "test.db.sqlite",
        "table_name": "customers",
        "prompt": "Filtrer les clients actifs",
        "extract": true/false  # Si true, extrait avec les filtres
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        if request.body:
            try:
                data = json.loads(request.body)
            except:
                data = request.POST.dict()
        else:
            data = request.POST.dict()
    except:
        data = request.POST.dict()

    db_type = data.get("db_type", "sqlite")
    db_name = data.get("db_name", "test.db.sqlite")
    table_name = data.get("table_name", "")
    prompt = data.get("prompt", "")
    should_extract = data.get("extract", False)

    if not table_name:
        return JsonResponse({"error": "table_name requis"}, status=400)
    
    if not prompt:
        return JsonResponse({"error": "Prompt requis"}, status=400)

    try:
        extractor = get_extractor(db_type, db_name)
    except Exception as e:
        return JsonResponse({"error": f"Connexion BDD échouée : {str(e)}"}, status=500)

    try:
        # Vérifier si les méthodes LLM existent
        if should_extract and hasattr(extractor, 'extract_with_llm_filters'):
            result = extractor.extract_with_llm_filters(table_name, f"{DATA_DIR}/filtered", prompt)
        elif hasattr(extractor, 'generate_filter_conditions'):
            result = extractor.generate_filter_conditions(table_name, prompt)
        else:
            # Fallback: méthode simple
            # Récupérer le schéma
            schema_df = extractor.extract_table_schema(table_name)
            schema = []
            if not schema_df.empty:
                for _, row in schema_df.iterrows():
                    schema.append(f"{row['COLUMN_NAME']} ({row['DATA_TYPE']})")
            
            # Filtres simulés
            filters = {
                "filters": [
                    {
                        "column": "id",
                        "operator": ">",
                        "value": "0",
                        "description": "Filtre par défaut"
                    }
                ],
                "logic": "AND",
                "explanation": "Filtres par défaut (méthode LLM non disponible)"
            }
            
            result = {
                "table": table_name,
                "schema": schema,
                "filters": filters,
                "llm_raw_response": "Méthode LLM non disponible",
                "prompt_used": prompt
            }
            
            if should_extract:
                # Exécuter une requête simple
                query = f"SELECT * FROM {table_name} LIMIT 100"
                try:
                    df = extractor.extract_data(query)
                    os.makedirs(f"{DATA_DIR}/filtered", exist_ok=True)
                    path = f"{DATA_DIR}/filtered/{table_name}_filtered.csv"
                    df.to_csv(path, index=False)
                    result["success"] = True
                    result["query"] = query
                    result["rows_extracted"] = len(df)
                    result["saved_to"] = path
                except Exception as e:
                    result["success"] = False
                    result["error"] = str(e)

        return JsonResponse(clean_for_json({
            "success": True,
            "result": result
        }))

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
def llm_analyze_database(request):
    """
    POST endpoint pour analyse complète par LLM
    Body: {
        "db_type": "sqlite",
        "db_name": "test.db.sqlite",
        "prompt": "Analyse les ventes du dernier trimestre"
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        if request.body:
            try:
                data = json.loads(request.body)
            except:
                data = request.POST.dict()
        else:
            data = request.POST.dict()
    except:
        data = request.POST.dict()

    db_type = data.get("db_type", "sqlite")
    db_name = data.get("db_name", "test.db.sqlite")
    prompt = data.get("prompt", "")

    if not prompt:
        return JsonResponse({"error": "Prompt requis"}, status=400)

    try:
        extractor = get_extractor(db_type, db_name)
    except Exception as e:
        return JsonResponse({"error": f"Connexion BDD échouée : {str(e)}"}, status=500)

    try:
        if hasattr(extractor, 'analyze_with_llm'):
            result = extractor.analyze_with_llm(prompt)
        else:
            # Fallback
            tables_df = extractor.get_all_tables()
            all_tables = tables_df['TABLE_NAME'].tolist() if not tables_df.empty else []
            
            # Récupérer les schémas
            schemas = {}
            for table in all_tables[:3]:
                try:
                    schema_df = extractor.extract_table_schema(table)
                    schemas[table] = [
                        f"{row['COLUMN_NAME']} ({row['DATA_TYPE']})" 
                        for _, row in schema_df.iterrows()
                    ][:5]
                except:
                    schemas[table] = ["Schema indisponible"]
            
            analysis = {
                "tables_to_extract": all_tables[:3],
                "filters": {
                    table: [{"column": "id", "operator": ">", "value": "0"}]
                    for table in all_tables[:2]
                },
                "overall_purpose": f"Analyse basée sur: {prompt}"
            }
            
            result = {
                "analysis": analysis,
                "all_tables": all_tables,
                "prompt": prompt,
                "llm_raw_response": "Méthode LLM non disponible - analyse par défaut"
            }

        return JsonResponse(clean_for_json({
            "success": True,
            "result": result
        }))

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def llm_interface(request):
    """Page d'interface pour le LLM"""
    return render(request, "llm_interface.html", {
        "page": "llm",
        "db_types": DB_TYPES,
    })