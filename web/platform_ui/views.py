import sys
import os
import pandas as pd
from sqlalchemy import text
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
 
# ajouter la racine du projet au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from django.shortcuts import render
from pipeline.orchestrator import Orchestrator
from django.shortcuts import render
from llm.llm_agent import LLMAgent

def home(request):

    return render(request, "home.html")


def analyze_quality(request):

    result = None

    if request.method == "POST":

        db_type = request.POST.get("db_type")
        db_name = request.POST.get("db_name")

        orchestrator = Orchestrator(db_type, db_name)

        result = orchestrator.run_quality_analysis()

    return render(request, "analyze.html", {"result": result})


def run_pipeline(request):

    result = None

    if request.method == "POST":

        db_type = request.POST.get("db_type")
        db_name = request.POST.get("db_name")

        orchestrator = Orchestrator(db_type, db_name)

        orchestrator.run_pipeline()

        result = "Pipeline executed successfully"

    return render(request, "run.html", {"result": result})

def llm_tables(request):

    result = None

    if request.method == "POST":

        tables = request.POST.get("tables")

        table_list = tables.split(",")

        llm = LLMAgent()

        result = llm.select_tables_for_cleaning(table_list)

    return render(request, "llm_tables.html", {"result": result})


    