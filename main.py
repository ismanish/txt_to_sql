import os
from typing import Annotated, Sequence, TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
import psycopg2
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, END
from db_inspector import DVDRentalInspector


load_dotenv()

inspector = DVDRentalInspector()
schema_info = inspector.get_schema_for_prompt()
table_stats = inspector.get_table_stats()