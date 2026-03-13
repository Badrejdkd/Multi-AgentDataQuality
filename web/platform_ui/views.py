import os
import sys
import glob
import json
import random
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import numpy as np
from django.http import JsonResponse

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
            return JsonResponse({
                "success": True,
                "mode":    "llm",
                "message": result,
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
        return JsonResponse({
            "success": True,
            "table":   table_name,
            "source":  source,
            "report":  clean_for_json(report),
        })
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
                "rows": 0, "quality_score": 0,
                "duplicates_removed": 0, "missing_filled": 0,
            })

    return JsonResponse({"success": True, "results": results, "db": db_name})