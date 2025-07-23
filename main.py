# enhanced_property_agent_with_predictive_analytics.py

import os
import re
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import hashlib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure Gemini API
api_key = st.secrets["gemini_key"]
genai.configure(api_key=api_key)

class QueryType(Enum):
    SQL_QUERY = "sql_query"
    TREND_ANALYSIS = "trend_analysis"  
    REPORT_GENERATION = "report_generation"
    FOLLOWUP_QUESTION = "followup_question"
    CLARIFICATION = "clarification"
    GENERAL_QUERY = "general_query"
    PREDICTIVE_ANALYSIS = "predictive_analysis"
    STRATEGIC_THINKING = "strategic_thinking"
    SCENARIO_PLANNING = "scenario_planning"

class PredictionType(Enum):
    OCCUPANCY_FORECAST = "occupancy_forecast"
    RENT_TREND = "rent_trend"
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    CASH_FLOW_PROJECTION = "cash_flow_projection"
    TENANT_CHURN = "tenant_churn"
    MARKET_ANALYSIS = "market_analysis"

@dataclass
class PredictiveInsight:
    """Represents a predictive insight or forecast"""
    prediction_type: PredictionType
    confidence_level: float  # 0-1 scale
    time_horizon: str  # e.g., "3 months", "1 year"
    key_metrics: Dict[str, Any]
    recommendations: List[str]
    risk_factors: List[str]
    data_quality: str  # "high", "medium", "low"

@dataclass
class ThinkingProcess:
    """Represents the AI's reasoning process"""
    problem_analysis: str
    data_assessment: str
    methodology: str
    assumptions: List[str]
    limitations: List[str]
    alternative_approaches: List[str]

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    timestamp: datetime
    user_query: str
    query_type: QueryType
    sql_generated: Optional[str]
    results: Optional[Dict[str, Any]]
    ai_response: str
    context_used: List[str]
    entities_mentioned: List[str]
    follow_up_suggestions: List[str]
    predictive_insights: Optional[PredictiveInsight] = None
    thinking_process: Optional[ThinkingProcess] = None

@dataclass
class ConversationMemory:
    """Maintains conversation context and memory"""
    session_id: str
    turns: List[ConversationTurn]
    current_context: Dict[str, Any]
    entity_references: Dict[str, Any]
    active_filters: Dict[str, Any]
    last_query_results: Optional[pd.DataFrame]
    conversation_summary: str
    learned_patterns: Dict[str, Any]  # Patterns the AI has learned about this property portfolio

class PredictiveAnalytics:
    """Advanced predictive analytics engine"""
    
    def __init__(self):
        self.seasonal_patterns = {}
        self.trend_cache = {}
    
    def analyze_occupancy_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze occupancy patterns and predict future trends"""
        analysis = {
            "current_occupancy": 0,
            "trend_direction": "stable",
            "seasonal_pattern": "unknown",
            "prediction_3_month": 0,
            "prediction_6_month": 0,
            "confidence": 0.5
        }
        
        try:
            if 'status' in df.columns:
                total_units = len(df)
                occupied_units = len(df[df['status'].str.lower().isin(['occupied', 'active'])])
                current_occupancy = (occupied_units / total_units) * 100 if total_units > 0 else 0
                analysis["current_occupancy"] = round(current_occupancy, 2)
                
                # Simple trend prediction based on current occupancy
                if current_occupancy >= 95:
                    analysis["trend_direction"] = "stable_high"
                    analysis["prediction_3_month"] = min(100, current_occupancy + 1)
                    analysis["prediction_6_month"] = min(100, current_occupancy + 2)
                elif current_occupancy <= 70:
                    analysis["trend_direction"] = "concerning"
                    analysis["prediction_3_month"] = max(0, current_occupancy - 2)
                    analysis["prediction_6_month"] = max(0, current_occupancy - 5)
                else:
                    analysis["trend_direction"] = "stable"
                    analysis["prediction_3_month"] = current_occupancy
                    analysis["prediction_6_month"] = current_occupancy + 1
                
                analysis["confidence"] = 0.7 if total_units > 10 else 0.5
        
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def predict_maintenance_needs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict future maintenance needs based on historical data"""
        prediction = {
            "high_priority_predicted": 0,
            "estimated_monthly_tickets": 0,
            "categories_at_risk": [],
            "preventive_recommendations": [],
            "confidence": 0.6
        }
        
        try:
            if 'priority' in df.columns and 'category' in df.columns:
                # Analyze current ticket distribution
                high_priority = len(df[df['priority'].str.lower() == 'emergency'])
                total_tickets = len(df)
                
                # Simple predictive model
                prediction["high_priority_predicted"] = max(1, int(high_priority * 1.2))
                prediction["estimated_monthly_tickets"] = max(5, int(total_tickets * 1.1))
                
                # Identify problematic categories
                if not df.empty:
                    category_counts = df['category'].value_counts()
                    top_categories = category_counts.head(3).index.tolist()
                    prediction["categories_at_risk"] = top_categories
                
                # Generate recommendations
                if high_priority > 0:
                    prediction["preventive_recommendations"].append(
                        "Implement preventive maintenance program for emergency-prone systems"
                    )
                if total_tickets > 20:
                    prediction["preventive_recommendations"].append(
                        "Consider increasing maintenance staff or outsourcing"
                    )
        
        except Exception as e:
            prediction["error"] = str(e)
        
        return prediction
    
    def forecast_cash_flow(self, payments_df: pd.DataFrame, leases_df: pd.DataFrame = None) -> Dict[str, Any]:
        """Forecast cash flow based on payment patterns"""
        forecast = {
            "projected_monthly_income": 0,
            "collection_rate_prediction": 95,
            "risk_assessment": "medium",
            "recommendations": [],
            "confidence": 0.6
        }
        
        try:
            if 'amount' in payments_df.columns:
                total_revenue = payments_df['amount'].sum()
                avg_monthly = total_revenue / 12 if len(payments_df) > 0 else 0
                
                # Calculate collection rate
                if 'paid_on' in payments_df.columns:
                    paid_payments = len(payments_df[payments_df['paid_on'].notna()])
                    total_payments = len(payments_df)
                    collection_rate = (paid_payments / total_payments) * 100 if total_payments > 0 else 95
                    
                    forecast["collection_rate_prediction"] = round(collection_rate, 1)
                    forecast["projected_monthly_income"] = round(avg_monthly * (collection_rate / 100), 2)
                    
                    if collection_rate >= 95:
                        forecast["risk_assessment"] = "low"
                    elif collection_rate >= 85:
                        forecast["risk_assessment"] = "medium"
                    else:
                        forecast["risk_assessment"] = "high"
                        forecast["recommendations"].append("Implement stricter collection procedures")
                
                if avg_monthly < 5000:
                    forecast["recommendations"].append("Consider rent adjustments or additional revenue streams")
        
        except Exception as e:
            forecast["error"] = str(e)
        
        return forecast

class StrategicThinking:
    """Strategic thinking and scenario planning engine"""
    
    def __init__(self):
        self.market_factors = [
            "local_economy", "interest_rates", "population_growth", 
            "competition", "regulatory_changes", "seasonal_demand"
        ]
    
    def analyze_market_position(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze current market position and strategic opportunities"""
        analysis = {
            "portfolio_strength": "unknown",
            "competitive_advantages": [],
            "growth_opportunities": [],
            "strategic_risks": [],
            "recommendations": []
        }
        
        try:
            # Analyze portfolio size and diversity
            if 'properties' in portfolio_data:
                props_df = portfolio_data['properties']
                property_count = len(props_df)
                
                if property_count >= 10:
                    analysis["portfolio_strength"] = "strong"
                    analysis["competitive_advantages"].append("Economies of scale in operations")
                elif property_count >= 5:
                    analysis["portfolio_strength"] = "moderate"
                else:
                    analysis["portfolio_strength"] = "developing"
                    analysis["growth_opportunities"].append("Portfolio expansion opportunities")
            
            # Analyze unit mix and occupancy
            if 'units' in portfolio_data:
                units_df = portfolio_data['units']
                if 'bedrooms' in units_df.columns:
                    unit_mix = units_df['bedrooms'].value_counts()
                    if len(unit_mix) > 3:
                        analysis["competitive_advantages"].append("Diverse unit mix appeals to multiple demographics")
                    
                if 'status' in units_df.columns:
                    vacancy_rate = len(units_df[units_df['status'] == 'vacant']) / len(units_df) * 100
                    if vacancy_rate > 10:
                        analysis["strategic_risks"].append("High vacancy rate indicates market or operational issues")
                        analysis["recommendations"].append("Conduct market analysis and review pricing strategy")
            
            # Analyze tenant relationships
            if 'tenants' in portfolio_data:
                tenants_df = portfolio_data['tenants']
                tenant_count = len(tenants_df)
                
                if tenant_count > 50:
                    analysis["competitive_advantages"].append("Large tenant base provides stable income stream")
                
                # Check for tenant concentration risk
                if 'leases' in portfolio_data:
                    leases_df = portfolio_data['leases']
                    if 'rent_amount' in leases_df.columns and len(leases_df) > 0:
                        avg_rent = leases_df['rent_amount'].mean()
                        max_rent = leases_df['rent_amount'].max()
                        if max_rent > avg_rent * 3:
                            analysis["strategic_risks"].append("High tenant concentration risk")
        
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def scenario_planning(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple scenarios and strategic responses"""
        scenarios = {
            "optimistic": {
                "description": "Strong market growth, high demand",
                "occupancy_change": "+5%",
                "rent_change": "+8%",
                "maintenance_change": "+2%",
                "strategy": "Expand portfolio, increase rents gradually"
            },
            "baseline": {
                "description": "Stable market conditions",
                "occupancy_change": "0%",
                "rent_change": "+3%",
                "maintenance_change": "+5%",
                "strategy": "Focus on operational efficiency and tenant retention"
            },
            "pessimistic": {
                "description": "Economic downturn, reduced demand",
                "occupancy_change": "-8%",
                "rent_change": "-2%",
                "maintenance_change": "+10%",
                "strategy": "Reduce costs, improve tenant value proposition, defer non-essential capex"
            }
        }
        
        return {
            "scenarios": scenarios,
            "recommended_preparations": [
                "Build cash reserves for at least 6 months operating expenses",
                "Diversify tenant base to reduce concentration risk",
                "Implement predictive maintenance to control costs",
                "Develop strong tenant relationships to improve retention"
            ],
            "key_indicators_to_monitor": [
                "Local employment rates",
                "New construction permits",
                "Average market rents",
                "Tenant payment patterns"
            ]
        }

class EntityExtractor:
    """Extracts and tracks entities mentioned in conversation"""
    
    def __init__(self):
        self.entity_patterns = {
            'tenant_names': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'property_names': r'\b[A-Z][a-zA-Z\s]+ (?:Apartments|Complex|Plaza|Tower|Building)\b',
            'addresses': r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)',
            'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            'amounts': r'\$[\d,]+(?:\.\d{2})?',
            'unit_numbers': r'\b(?:Unit|Apt|Apartment)\s*#?\s*\d+[A-Z]?\b'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {}
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))
        return entities

class ContextResolver:
    """Resolves pronouns and references using conversation context"""
    
    def __init__(self):
        self.reference_patterns = {
            'they': ['tenants', 'properties', 'units', 'payments'],
            'them': ['tenants', 'properties', 'units', 'payments'],
            'those': ['tenants', 'properties', 'units', 'tickets'],
            'these': ['tenants', 'properties', 'units', 'tickets'],
            'it': ['property', 'unit', 'payment', 'ticket'],
            'that': ['property', 'unit', 'payment', 'ticket']
        }
    
    def resolve_references(self, query: str, memory: ConversationMemory) -> str:
        """Resolve pronouns and references in the query"""
        resolved_query = query.lower()
        
        # Get the last few turns for context
        recent_turns = memory.turns[-3:] if len(memory.turns) >= 3 else memory.turns
        
        for pronoun, possible_entities in self.reference_patterns.items():
            if pronoun in resolved_query:
                resolved_entity = self._find_recent_entity(recent_turns, possible_entities)
                if resolved_entity:
                    resolved_query = resolved_query.replace(pronoun, resolved_entity)
        
        # Handle specific follow-up patterns
        if any(phrase in resolved_query for phrase in ['who are they', 'what are they', 'show me them']):
            if memory.last_query_results is not None and not memory.last_query_results.empty:
                columns = memory.last_query_results.columns.tolist()
                if any(col in ['first_name', 'last_name', 'tenant_name'] for col in columns):
                    resolved_query = "show me the detailed information for these tenants"
                elif any(col in ['property_name', 'address'] for col in columns):
                    resolved_query = "show me the detailed information for these properties"
                elif any(col in ['unit_number', 'unit_id'] for col in columns):
                    resolved_query = "show me the detailed information for these units"
        
        return resolved_query
    
    def _find_recent_entity(self, recent_turns: List[ConversationTurn], possible_entities: List[str]) -> Optional[str]:
        """Find the most recently mentioned relevant entity"""
        for turn in reversed(recent_turns):
            for entity in possible_entities:
                if entity in turn.user_query.lower() or entity in turn.ai_response.lower():
                    return entity
        return None

class PropertyRAGSystem:
    """Enhanced RAG system with conversation memory and predictive knowledge"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.vector_store = None
        self.documents = []
        self.conversation_embeddings = {}
        self._initialize_domain_knowledge()
    
    def _initialize_domain_knowledge(self):
        """Initialize with enhanced property management domain knowledge including predictive analytics"""
        domain_docs = [
            """Property Management Best Practices:
            - Regular property inspections should be conducted quarterly
            - Rent collection should be automated with late fees after 5 days
            - Maintenance requests should be categorized by urgency: Emergency (24hrs), Urgent (3 days), Standard (7 days)
            - Tenant screening should include credit check, employment verification, and references
            - Security deposits typically equal 1-2 months rent depending on local laws
            - Monthly rent-to-income ratio should not exceed 30% for qualified tenants""",
            
            """Predictive Analytics for Property Management:
            - Occupancy forecasting uses seasonal patterns, local economic indicators, and historical data
            - Maintenance prediction models analyze equipment age, usage patterns, and failure history
            - Rent optimization considers market comparables, tenant quality, and demand patterns
            - Cash flow forecasting incorporates collection rates, vacancy predictions, and expense trends
            - Tenant churn prediction uses payment history, lease terms, and satisfaction indicators
            - Market analysis includes demographic trends, employment data, and development patterns""",
            
            """Strategic Planning Frameworks:
            - Portfolio diversification reduces concentration risk across property types and locations
            - Capital allocation should prioritize high-ROI improvements and strategic acquisitions
            - Risk management includes insurance coverage, reserve funds, and scenario planning
            - Market positioning analysis compares amenities, pricing, and service levels
            - Growth strategies may include acquisition, development, or value-add improvements
            - Technology adoption can improve operational efficiency and tenant satisfaction""",
            
            """Key Performance Indicators and Benchmarks:
            - Occupancy rate targets: 95-98% for stabilized properties
            - Rent collection rate: 98%+ considered excellent, 95%+ acceptable
            - Tenant turnover: <10% annually for residential, varies by market
            - Maintenance cost ratio: 15-25% of gross rental income
            - Net Operating Income (NOI) margins: 60-80% for well-managed properties
            - Cash-on-cash returns: 8-12% typical for investment properties""",
            
            """Common SQL Patterns for Analytics:
            - Occupancy trends: Track status changes over time with date filters
            - Payment patterns: Analyze collection rates, late payments, and seasonal variations
            - Maintenance forecasting: Group tickets by category, priority, and resolution time
            - Tenant lifecycle: Track move-ins, renewals, and move-outs by period
            - Revenue optimization: Compare actual vs market rents, identify opportunities
            - Cost analysis: Track maintenance, utility, and operational costs by property"""
        ]
        
        # Create document objects
        for i, doc in enumerate(domain_docs):
            chunks = self.text_splitter.split_text(doc)
            for j, chunk in enumerate(chunks):
                self.documents.append(Document(
                    page_content=chunk,
                    metadata={"source": f"domain_knowledge_{i}_{j}", "type": "domain"}
                ))
        
        self._build_vector_store()
    
    def _build_vector_store(self):
        """Build FAISS vector store from documents"""
        if not self.documents:
            return
        
        texts = [doc.page_content for doc in self.documents]
        embeddings = self.embedding_model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.vector_store = faiss.IndexFlatL2(dimension)
        self.vector_store.add(embeddings.astype('float32'))
    
    def add_conversation_turn(self, turn: ConversationTurn):
        """Add conversation turn to knowledge base"""
        turn_text = f"""
        User Query: {turn.user_query}
        Query Type: {turn.query_type.value}
        SQL Generated: {turn.sql_generated or 'None'}
        AI Response: {turn.ai_response[:500]}...
        Entities: {', '.join(turn.entities_mentioned)}
        """
        
        chunks = self.text_splitter.split_text(turn_text)
        for chunk in chunks:
            self.documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": "conversation_history", 
                    "timestamp": turn.timestamp.isoformat(),
                    "type": "conversation"
                }
            ))
        
        # Store conversation embedding for quick retrieval
        turn_embedding = self.embedding_model.encode([turn.user_query + " " + turn.ai_response])
        self.conversation_embeddings[turn.timestamp.isoformat()] = turn_embedding[0]
        
        # Rebuild vector store (in production, use incremental updates)
        self._build_vector_store()
    
    def retrieve_context(self, query: str, memory: ConversationMemory, k: int = 5) -> Tuple[List[str], List[str]]:
        """Retrieve relevant context including conversation history"""
        if not self.vector_store:
            return [], []
        
        # Get embeddings for query
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.vector_store.search(query_embedding.astype('float32'), k)
        
        domain_context = []
        conversation_context = []
        
        for idx in indices[0]:
            if idx < len(self.documents):
                doc = self.documents[idx]
                if doc.metadata.get("type") == "domain":
                    domain_context.append(doc.page_content)
                elif doc.metadata.get("type") == "conversation":
                    conversation_context.append(doc.page_content)
        
        # Also include recent conversation turns
        recent_turns = memory.turns[-3:] if len(memory.turns) >= 3 else memory.turns
        for turn in recent_turns:
            if turn.timestamp not in [t.timestamp for t in memory.turns[-1:]]:
                turn_context = f"Previous: {turn.user_query} -> {turn.ai_response[:200]}..."
                conversation_context.append(turn_context)
        
        return domain_context, conversation_context

# Session Management Functions (same as before)
def save_sessions_to_disk(agent, filepath: str = "sessions.json"):
    """Save all sessions to disk for persistence"""
    sessions_data = {}
    for session_id, memory in agent.memory_store.items():
        sessions_data[session_id] = {
            "session_id": memory.session_id,
            "conversation_summary": memory.conversation_summary,
            "created_at": memory.turns[0].timestamp.isoformat() if memory.turns else datetime.now().isoformat(),
            "last_activity": memory.turns[-1].timestamp.isoformat() if memory.turns else datetime.now().isoformat(),
            "turn_count": len(memory.turns),
            "learned_patterns": memory.learned_patterns,
            "turns": [
                {
                    "timestamp": turn.timestamp.isoformat(),
                    "user_query": turn.user_query,
                    "query_type": turn.query_type.value,
                    "sql_generated": turn.sql_generated,
                    "ai_response": turn.ai_response,
                    "entities_mentioned": turn.entities_mentioned,
                    "follow_up_suggestions": turn.follow_up_suggestions
                }
                for turn in memory.turns
            ],
            "current_context": memory.current_context,
            "entity_references": memory.entity_references,
            "active_filters": memory.active_filters
        }
    
    with open(filepath, 'w') as f:
        json.dump(sessions_data, f, indent=2)

def load_sessions_from_disk(agent, filepath: str = "sessions.json"):
    """Load sessions from disk"""
    if not Path(filepath).exists():
        return
    
    try:
        with open(filepath, 'r') as f:
            sessions_data = json.load(f)
        
        for session_id, session_info in sessions_data.items():
            memory = ConversationMemory(
                session_id=session_id,
                turns=[],
                current_context=session_info.get("current_context", {}),
                entity_references=session_info.get("entity_references", {}),
                active_filters=session_info.get("active_filters", {}),
                last_query_results=None,
                conversation_summary=session_info.get("conversation_summary", ""),
                learned_patterns=session_info.get("learned_patterns", {})
            )
            
            for turn_data in session_info.get("turns", []):
                turn = ConversationTurn(
                    timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                    user_query=turn_data["user_query"],
                    query_type=QueryType(turn_data["query_type"]),
                    sql_generated=turn_data.get("sql_generated"),
                    results=None,
                    ai_response=turn_data["ai_response"],
                    context_used=[],
                    entities_mentioned=turn_data.get("entities_mentioned", []),
                    follow_up_suggestions=turn_data.get("follow_up_suggestions", [])
                )
                memory.turns.append(turn)
            
            agent.memory_store[session_id] = memory
    except Exception as e:
        st.error(f"Error loading sessions: {e}")

def get_session_summary(memory: ConversationMemory) -> dict:
    """Get a summary of a session for display"""
    if not memory.turns:
        return {
            "title": "Empty Session",
            "last_activity": "No activity",
            "question_count": 0,
            "preview": "No questions asked"
        }
    
    first_question = memory.turns[0].user_query[:50] + "..." if len(memory.turns[0].user_query) > 50 else memory.turns[0].user_query
    last_activity = memory.turns[-1].timestamp.strftime("%Y-%m-%d %H:%M")
    
    return {
        "title": f"Session: {first_question}",
        "last_activity": last_activity,
        "question_count": len(memory.turns),
        "preview": f"Last: {memory.turns[-1].user_query[:40]}..." if len(memory.turns[-1].user_query) > 40 else memory.turns[-1].user_query
    }

class PropertyManagementAgent:
    """Enhanced agentic AI system with predictive analytics and strategic thinking"""
    
    def __init__(self, rag_system: PropertyRAGSystem):
        self.rag_system = rag_system
        self.entity_extractor = EntityExtractor()
        self.context_resolver = ContextResolver()
        self.predictive_engine = PredictiveAnalytics()
        self.strategic_engine = StrategicThinking()
        self.memory_store = {}
        
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        return """
        You are an expert Property Management AI Agent with advanced memory, predictive analytics, and strategic thinking capabilities.
        
        CORE CAPABILITIES:
        - Generate and execute SQL queries for property management databases
        - Maintain conversation context and memory across multiple turns
        - Perform predictive analytics and forecasting
        - Conduct strategic analysis and scenario planning
        - Provide data-driven insights and recommendations
        
        PREDICTIVE ANALYTICS:
        - Occupancy forecasting based on historical patterns
        - Maintenance prediction using equipment and category analysis
        - Cash flow projections with risk assessment
        - Market trend analysis and competitive positioning
        - Tenant churn prediction and retention strategies
        
        STRATEGIC THINKING:
        - Portfolio optimization and diversification analysis
        - Growth opportunity identification
        - Risk assessment and mitigation planning
        - Scenario planning for different market conditions
        - Investment prioritization and resource allocation
        
        DATABASE SCHEMA:
        -- tenants(id, first_name, last_name, email, phone, date_of_birth, created_at)
        -- properties(id, name, address_line1, address_line2, city, state, postal_code, country, created_at)
        -- units(id, property_id, unit_number, floor, bedrooms, bathrooms, square_feet, status, created_at)
        -- rooms(id, unit_id, room_name, room_type, size_sq_ft, status, created_at)
        -- agents(id, first_name, last_name, email, phone, created_at)
        -- leases(id, tenant_id, room_id, agent_id, start_date, end_date, rent_amount, security_deposit, status, created_at)
        -- maintenance_tickets(id, room_id, unit_id, subcategory, scheduled_for, completed_on, created_at)
        -- complaint_tickets(id, lease_id, severity, complaint_type, filed_on, resolved_on, resolution)
        -- payments(id, lease_id, payment_type, transaction_type, due_date, amount, method, paid_on, created_at)
        -- chat_rooms(id, tenant_id, created_at, last_updated, status)
        -- conversation_messages(id, chat_room_id, author_type, author_id, message_text, sent_at)
        
        RESPONSE GUIDELINES:
        - Always explain your reasoning process for complex analyses
        - Provide confidence levels for predictions (high/medium/low)
        - Include assumptions and limitations in your analysis
        - Suggest actionable recommendations based on insights
        - Consider multiple scenarios when appropriate
        - Use data visualization suggestions when helpful
        
        When performing predictive analysis, always:
        1. Explain your methodology
        2. State your assumptions clearly
        3. Provide confidence intervals or levels
        4. Suggest data improvements for better accuracy
        5. Include risk factors and limitations
        """
    
    def get_or_create_memory(self, session_id: str) -> ConversationMemory:
        """Get existing memory or create new one for session"""
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ConversationMemory(
                session_id=session_id,
                turns=[],
                current_context={},
                entity_references={},
                active_filters={},
                last_query_results=None,
                conversation_summary="",
                learned_patterns={}
            )
        return self.memory_store[session_id]
    
    def process_query(self, user_query: str, db_path: str, session_id: str = "default") -> Dict[str, Any]:
        """Main processing pipeline with predictive analytics and strategic thinking"""
        
        # Get conversation memory
        memory = self.get_or_create_memory(session_id)
        
        # Resolve references using context
        resolved_query = self.context_resolver.resolve_references(user_query, memory)
        
        # Extract entities
        entities = self.entity_extractor.extract_entities(resolved_query)
        entity_list = []
        for entity_type, entity_values in entities.items():
            entity_list.extend(entity_values)
        
        # Retrieve relevant context
        domain_context, conversation_context = self.rag_system.retrieve_context(resolved_query, memory)
        
        # Classify query type with enhanced predictive/strategic detection
        query_type = self._classify_intent(resolved_query, memory)
        
        # Process based on type
        if query_type == QueryType.PREDICTIVE_ANALYSIS:
            result = self._handle_predictive_analysis(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.STRATEGIC_THINKING:
            result = self._handle_strategic_thinking(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.SCENARIO_PLANNING:
            result = self._handle_scenario_planning(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.FOLLOWUP_QUESTION:
            result = self._handle_followup_question(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.SQL_QUERY:
            result = self._handle_sql_query(resolved_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.TREND_ANALYSIS:
            result = self._handle_trend_analysis(resolved_query, db_path, memory, domain_context, conversation_context)
        else:
            result = self._handle_general_query(resolved_query, db_path, memory, domain_context, conversation_context)
        
        # Create conversation turn with predictive insights
        turn = ConversationTurn(
            timestamp=datetime.now(),
            user_query=user_query,
            query_type=query_type,
            sql_generated=result.get('sql'),
            results=result.get('results'),
            ai_response=result.get('response', ''),
            context_used=domain_context + conversation_context,
            entities_mentioned=entity_list,
            follow_up_suggestions=result.get('follow_up_suggestions', []),
            predictive_insights=result.get('predictive_insights'),
            thinking_process=result.get('thinking_process')
        )
        
        # Update memory
        memory.turns.append(turn)
        if 'results' in result and isinstance(result['results'], pd.DataFrame):
            memory.last_query_results = result['results']
        
        # Learn patterns for future predictions
        self._learn_patterns(turn, memory)
        
        # Add to RAG system
        self.rag_system.add_conversation_turn(turn)
        
        return result
    
    def _classify_intent(self, query: str, memory: ConversationMemory) -> QueryType:
        """Enhanced intent classification including predictive and strategic queries"""
        
        # Keywords for predictive analysis
        predictive_keywords = [
            'predict', 'forecast', 'projection', 'future', 'trend', 'expected',
            'will be', 'anticipate', 'estimate', 'likely', 'next month', 'next year',
            'upcoming', 'outlook', 'what if', 'scenario'
        ]
        
        # Keywords for strategic thinking
        strategic_keywords = [
            'strategy', 'strategic', 'planning', 'growth', 'opportunity', 'risk',
            'competitive', 'market position', 'portfolio', 'investment', 'optimize',
            'improve', 'recommendation', 'should we', 'best approach', 'options'
        ]
        
        # Keywords for scenario planning
        scenario_keywords = [
            'scenario', 'what if', 'alternatives', 'different outcomes', 'worst case',
            'best case', 'contingency', 'prepare for', 'multiple options', 'compare options'
        ]
        
        query_lower = query.lower()
        
        # Check for predictive analysis
        if any(keyword in query_lower for keyword in predictive_keywords):
            return QueryType.PREDICTIVE_ANALYSIS
        
        # Check for strategic thinking
        if any(keyword in query_lower for keyword in strategic_keywords):
            return QueryType.STRATEGIC_THINKING
        
        # Check for scenario planning
        if any(keyword in query_lower for keyword in scenario_keywords):
            return QueryType.SCENARIO_PLANNING
        
        # Check for obvious follow-up patterns
        followup_indicators = [
            'who are they', 'what are they', 'show me them', 'show them', 
            'tell me more', 'more details', 'expand on that', 'show me more',
            'which ones', 'what about', 'how about', 'and them', 'those too'
        ]
        
        if any(indicator in query_lower for indicator in followup_indicators):
            return QueryType.FOLLOWUP_QUESTION
        
        # Check if this relates to previous query results
        if memory.last_query_results is not None and len(memory.turns) > 0:
            last_turn = memory.turns[-1]
            
            # Common follow-up patterns after count queries
            if 'count' in last_turn.sql_generated.lower() if last_turn.sql_generated else False:
                detail_words = ['details', 'names', 'list', 'show', 'who', 'which', 'what']
                if any(word in query_lower for word in detail_words):
                    return QueryType.FOLLOWUP_QUESTION
        
        # Standard classification
        return QueryType.SQL_QUERY
    
    def _handle_predictive_analysis(self, resolved_query: str, original_query: str, db_path: str, 
                                  memory: ConversationMemory, domain_context: List[str], 
                                  conversation_context: List[str]) -> Dict[str, Any]:
        """Handle predictive analysis queries"""
        
        try:
            # First, get relevant data
            data_query = self._generate_predictive_data_query(resolved_query)
            
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(data_query, conn)
            
            # Get additional context data
            portfolio_data = self._get_portfolio_data(conn)
            conn.close()
            
            # Determine prediction type
            prediction_type = self._determine_prediction_type(resolved_query)
            
            # Perform predictive analysis
            if prediction_type == PredictionType.OCCUPANCY_FORECAST:
                prediction_result = self.predictive_engine.analyze_occupancy_trends(df)
                analysis_type = "Occupancy Forecast"
            elif prediction_type == PredictionType.MAINTENANCE_PREDICTION:
                prediction_result = self.predictive_engine.predict_maintenance_needs(df)
                analysis_type = "Maintenance Prediction"
            elif prediction_type == PredictionType.CASH_FLOW_PROJECTION:
                leases_df = portfolio_data.get('leases', pd.DataFrame())
                prediction_result = self.predictive_engine.forecast_cash_flow(df, leases_df)
                analysis_type = "Cash Flow Forecast"
            else:
                prediction_result = {"message": "General predictive analysis performed"}
                analysis_type = "Predictive Analysis"
            
            # Generate thinking process
            thinking_process = ThinkingProcess(
                problem_analysis=f"Analyzing {original_query} requires predictive modeling based on historical data patterns.",
                data_assessment=f"Used {len(df)} records from the database. Data quality: {'Good' if len(df) > 10 else 'Limited'}",
                methodology=f"Applied statistical analysis and pattern recognition for {prediction_type.value if hasattr(prediction_type, 'value') else 'general'} prediction.",
                assumptions=[
                    "Historical patterns will continue",
                    "No major market disruptions",
                    "Current operational practices maintained"
                ],
                limitations=[
                    "Limited historical data may affect accuracy",
                    "External factors not fully accounted for",
                    "Model assumes stable market conditions"
                ],
                alternative_approaches=[
                    "Machine learning models with more data",
                    "External market data integration",
                    "Real-time monitoring and adjustment"
                ]
            )
            
            # Create predictive insight
            confidence = prediction_result.get('confidence', 0.6)
            predictive_insight = PredictiveInsight(
                prediction_type=prediction_type,
                confidence_level=confidence,
                time_horizon="3-12 months",
                key_metrics=prediction_result,
                recommendations=prediction_result.get('recommendations', []),
                risk_factors=prediction_result.get('risk_factors', []),
                data_quality="good" if len(df) > 20 else "limited"
            )
            
            # Generate comprehensive response
            response = self._generate_predictive_response(
                original_query, analysis_type, prediction_result, thinking_process, confidence
            )
            
            # Generate follow-up suggestions
            suggestions = [
                "What are the key risk factors?",
                "How can we improve prediction accuracy?",
                "What actions should we take based on this forecast?",
                "Show me the underlying data trends"
            ]
            
            return {
                "type": "predictive_analysis",
                "sql": data_query,
                "results": df,
                "response": response,
                "predictive_insights": predictive_insight,
                "thinking_process": thinking_process,
                "follow_up_suggestions": suggestions,
                "success": True,
                "analysis_type": analysis_type
            }
            
        except Exception as e:
            return {
                "type": "predictive_analysis",
                "error": str(e),
                "response": f"I encountered an error while performing predictive analysis: {str(e)}",
                "success": False
            }
    
    def _handle_strategic_thinking(self, resolved_query: str, original_query: str, db_path: str, 
                                 memory: ConversationMemory, domain_context: List[str], 
                                 conversation_context: List[str]) -> Dict[str, Any]:
        """Handle strategic thinking and planning queries"""
        
        try:
            # Get comprehensive portfolio data
            conn = sqlite3.connect(db_path)
            portfolio_data = self._get_portfolio_data(conn)
            conn.close()
            
            # Perform strategic analysis
            strategic_analysis = self.strategic_engine.analyze_market_position(portfolio_data)
            
            # Generate thinking process
            thinking_process = ThinkingProcess(
                problem_analysis=f"Strategic analysis of '{original_query}' requires comprehensive portfolio assessment and market positioning.",
                data_assessment=f"Analyzed {len(portfolio_data)} data categories including properties, units, tenants, and financial metrics.",
                methodology="Applied strategic management frameworks including portfolio analysis, competitive positioning, and growth assessment.",
                assumptions=[
                    "Current market conditions remain relatively stable",
                    "Historical performance indicates future potential",
                    "Strategic initiatives can be effectively executed"
                ],
                limitations=[
                    "Limited external market data",
                    "Strategic recommendations require market validation",
                    "Implementation depends on available resources"
                ],
                alternative_approaches=[
                    "External market research and benchmarking",
                    "Stakeholder interviews and feedback",
                    "Pilot testing of strategic initiatives"
                ]
            )
            
            # Generate strategic response
            response = self._generate_strategic_response(original_query, strategic_analysis, thinking_process)
            
            # Generate strategic follow-up suggestions
            suggestions = [
                "What are our biggest competitive advantages?",
                "Which growth opportunities should we prioritize?",
                "How do we mitigate the identified risks?",
                "What investments would give us the highest ROI?"
            ]
            
            return {
                "type": "strategic_thinking",
                "results": pd.DataFrame([strategic_analysis]),  # Convert to DataFrame for display
                "response": response,
                "thinking_process": thinking_process,
                "strategic_analysis": strategic_analysis,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "strategic_thinking",
                "error": str(e),
                "response": f"I encountered an error during strategic analysis: {str(e)}",
                "success": False
            }
    
    def _handle_scenario_planning(self, resolved_query: str, original_query: str, db_path: str, 
                                memory: ConversationMemory, domain_context: List[str], 
                                conversation_context: List[str]) -> Dict[str, Any]:
        """Handle scenario planning and what-if analysis"""
        
        try:
            # Get current metrics
            conn = sqlite3.connect(db_path)
            portfolio_data = self._get_portfolio_data(conn)
            conn.close()
            
            # Calculate current baseline metrics
            current_metrics = self._calculate_baseline_metrics(portfolio_data)
            
            # Generate scenarios
            scenario_analysis = self.strategic_engine.scenario_planning(current_metrics)
            
            # Generate thinking process
            thinking_process = ThinkingProcess(
                problem_analysis=f"Scenario planning for '{original_query}' requires modeling multiple future outcomes and their implications.",
                data_assessment="Used current portfolio metrics as baseline for scenario modeling.",
                methodology="Applied scenario planning methodology with optimistic, baseline, and pessimistic cases.",
                assumptions=[
                    "Each scenario represents a plausible future state",
                    "Key variables change proportionally",
                    "Strategic responses can be implemented effectively"
                ],
                limitations=[
                    "Scenarios are simplified models of complex realities",
                    "Black swan events not accounted for",
                    "Interdependencies between variables may be underestimated"
                ],
                alternative_approaches=[
                    "Monte Carlo simulation for probability distributions",
                    "Dynamic scenario modeling with feedback loops",
                    "Real options analysis for strategic flexibility"
                ]
            )
            
            # Generate scenario response
            response = self._generate_scenario_response(original_query, scenario_analysis, current_metrics, thinking_process)
            
            # Convert scenarios to DataFrame for display
            scenarios_df = pd.DataFrame(scenario_analysis['scenarios']).T
            
            suggestions = [
                "Which scenario is most likely?",
                "How should we prepare for the worst case?",
                "What early indicators should we monitor?",
                "Which strategies work across all scenarios?"
            ]
            
            return {
                "type": "scenario_planning",
                "results": scenarios_df,
                "response": response,
                "thinking_process": thinking_process,
                "scenario_analysis": scenario_analysis,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "scenario_planning",
                "error": str(e),
                "response": f"I encountered an error during scenario planning: {str(e)}",
                "success": False
            }
    
    def _generate_predictive_data_query(self, query: str) -> str:
        """Generate SQL query to get data needed for prediction"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['occupancy', 'vacancy', 'units', 'occupied']):
            return "SELECT * FROM units ORDER BY created_at DESC"
        elif any(word in query_lower for word in ['maintenance', 'tickets', 'repairs']):
            return "SELECT * FROM service_tickets ORDER BY created_at DESC"
        elif any(word in query_lower for word in ['payment', 'rent', 'income', 'cash flow']):
            return "SELECT * FROM payments ORDER BY created_at DESC"
        elif any(word in query_lower for word in ['tenant', 'churn', 'turnover']):
            return "SELECT * FROM tenants ORDER BY created_at DESC"
        else:
            return "SELECT * FROM payments ORDER BY created_at DESC LIMIT 100"
    
    def _get_portfolio_data(self, conn) -> Dict[str, pd.DataFrame]:
        """Get comprehensive portfolio data for analysis"""
        portfolio_data = {}
        
        tables = ['properties', 'units', 'tenants', 'leases', 'payments', 'service_tickets']
        
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                portfolio_data[table] = df
            except:
                portfolio_data[table] = pd.DataFrame()
        
        return portfolio_data
    
    def _determine_prediction_type(self, query: str) -> PredictionType:
        """Determine the type of prediction needed"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['occupancy', 'vacancy', 'units']):
            return PredictionType.OCCUPANCY_FORECAST
        elif any(word in query_lower for word in ['maintenance', 'repair', 'tickets']):
            return PredictionType.MAINTENANCE_PREDICTION
        elif any(word in query_lower for word in ['cash flow', 'income', 'revenue']):
            return PredictionType.CASH_FLOW_PROJECTION
        elif any(word in query_lower for word in ['tenant', 'churn', 'turnover']):
            return PredictionType.TENANT_CHURN
        elif any(word in query_lower for word in ['rent', 'pricing', 'market']):
            return PredictionType.RENT_TREND
        else:
            return PredictionType.MARKET_ANALYSIS
    
    def _calculate_baseline_metrics(self, portfolio_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate current baseline metrics for scenario planning"""
        metrics = {}
        
        try:
            if 'units' in portfolio_data and not portfolio_data['units'].empty:
                units_df = portfolio_data['units']
                total_units = len(units_df)
                occupied_units = len(units_df[units_df['status'].str.lower().isin(['occupied', 'active'])])
                metrics['occupancy_rate'] = (occupied_units / total_units) * 100 if total_units > 0 else 0
                metrics['total_units'] = total_units
            
            if 'payments' in portfolio_data and not portfolio_data['payments'].empty:
                payments_df = portfolio_data['payments']
                metrics['monthly_revenue'] = payments_df['amount'].sum() / 12 if len(payments_df) > 0 else 0
                
                if 'paid_on' in payments_df.columns:
                    paid_count = len(payments_df[payments_df['paid_on'].notna()])
                    total_count = len(payments_df)
                    metrics['collection_rate'] = (paid_count / total_count) * 100 if total_count > 0 else 95
            
            if 'service_tickets' in portfolio_data and not portfolio_data['service_tickets'].empty:
                tickets_df = portfolio_data['service_tickets']
                metrics['monthly_tickets'] = len(tickets_df) / 12 if len(tickets_df) > 0 else 0
                
                if 'priority' in tickets_df.columns:
                    emergency_count = len(tickets_df[tickets_df['priority'].str.lower() == 'emergency'])
                    metrics['emergency_rate'] = (emergency_count / len(tickets_df)) * 100 if len(tickets_df) > 0 else 5
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _generate_predictive_response(self, query: str, analysis_type: str, 
                                    prediction_result: Dict[str, Any], 
                                    thinking_process: ThinkingProcess, 
                                    confidence: float) -> str:
        """Generate a comprehensive predictive analysis response"""
        
        confidence_text = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "moderate"
        
        response = f"""## 🔮 {analysis_type}

**My Analysis Process:**
{thinking_process.problem_analysis}

**Data Assessment:**
{thinking_process.data_assessment}

**Methodology:**
{thinking_process.methodology}

### 📊 Key Predictions

"""
        
        # Add specific predictions based on type
        for key, value in prediction_result.items():
            if key not in ['error', 'confidence', 'recommendations']:
                if isinstance(value, (int, float)):
                    response += f"- **{key.replace('_', ' ').title()}:** {value:,.2f}\n"
                elif isinstance(value, list) and value:
                    response += f"- **{key.replace('_', ' ').title()}:** {', '.join(map(str, value))}\n"
                elif isinstance(value, str):
                    response += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        response += f"""
### 🎯 Confidence Level: {confidence_text.title()} ({confidence:.1%})

**Key Assumptions:**
"""
        for assumption in thinking_process.assumptions:
            response += f"- {assumption}\n"
        
        response += "\n**Important Limitations:**\n"
        for limitation in thinking_process.limitations:
            response += f"- {limitation}\n"
        
        if 'recommendations' in prediction_result and prediction_result['recommendations']:
            response += "\n### 💡 Recommended Actions:\n"
            for rec in prediction_result['recommendations']:
                response += f"- {rec}\n"
        
        return response
    
    def _generate_strategic_response(self, query: str, strategic_analysis: Dict[str, Any], 
                                   thinking_process: ThinkingProcess) -> str:
        """Generate a comprehensive strategic analysis response"""
        
        response = f"""## 🎯 Strategic Analysis

**Strategic Question:** {query}

**My Thinking Process:**
{thinking_process.problem_analysis}

### 📈 Portfolio Assessment

**Current Position:** {strategic_analysis.get('portfolio_strength', 'Unknown')}

"""
        
        if strategic_analysis.get('competitive_advantages'):
            response += "**Competitive Advantages:**\n"
            for advantage in strategic_analysis['competitive_advantages']:
                response += f"- ✅ {advantage}\n"
            response += "\n"
        
        if strategic_analysis.get('growth_opportunities'):
            response += "**Growth Opportunities:**\n"
            for opportunity in strategic_analysis['growth_opportunities']:
                response += f"- 🚀 {opportunity}\n"
            response += "\n"
        
        if strategic_analysis.get('strategic_risks'):
            response += "**Strategic Risks:**\n"
            for risk in strategic_analysis['strategic_risks']:
                response += f"- ⚠️ {risk}\n"
            response += "\n"
        
        if strategic_analysis.get('recommendations'):
            response += "### 💡 Strategic Recommendations:\n"
            for rec in strategic_analysis['recommendations']:
                response += f"- {rec}\n"
        
        response += f"""
### 🤔 Methodology & Considerations

**Analysis Framework:** {thinking_process.methodology}

**Key Assumptions:**
"""
        for assumption in thinking_process.assumptions:
            response += f"- {assumption}\n"
        
        response += "\n**Alternative Approaches to Consider:**\n"
        for alternative in thinking_process.alternative_approaches:
            response += f"- {alternative}\n"
        
        return response
    
    def _generate_scenario_response(self, query: str, scenario_analysis: Dict[str, Any], 
                                  current_metrics: Dict[str, Any], 
                                  thinking_process: ThinkingProcess) -> str:
        """Generate a comprehensive scenario planning response"""
        
        response = f"""## 🎲 Scenario Planning Analysis

**Planning Question:** {query}

**Current Baseline Metrics:**
"""
        
        for metric, value in current_metrics.items():
            if metric != 'error':
                response += f"- **{metric.replace('_', ' ').title()}:** {value:,.2f}\n"
        
        response += f"""
### 📊 Three Scenarios Analyzed

**My Methodology:** {thinking_process.methodology}

"""
        
        scenarios = scenario_analysis.get('scenarios', {})
        for scenario_name, scenario_data in scenarios.items():
            response += f"""
**{scenario_name.title()} Scenario:**
- *{scenario_data.get('description', 'No description')}*
- Occupancy Impact: {scenario_data.get('occupancy_change', 'N/A')}
- Rent Impact: {scenario_data.get('rent_change', 'N/A')}
- Maintenance Impact: {scenario_data.get('maintenance_change', 'N/A')}
- **Recommended Strategy:** {scenario_data.get('strategy', 'No strategy defined')}

"""
        
        response += "### 🛡️ Preparation Strategies\n"
        for prep in scenario_analysis.get('recommended_preparations', []):
            response += f"- {prep}\n"
        
        response += "\n### 📊 Key Indicators to Monitor\n"
        for indicator in scenario_analysis.get('key_indicators_to_monitor', []):
            response += f"- {indicator}\n"
        
        response += f"""
### 🤔 Analysis Considerations

**Key Assumptions:**
"""
        for assumption in thinking_process.assumptions:
            response += f"- {assumption}\n"
        
        response += "\n**Limitations of This Analysis:**\n"
        for limitation in thinking_process.limitations:
            response += f"- {limitation}\n"
        
        return response
    
    def _learn_patterns(self, turn: ConversationTurn, memory: ConversationMemory):
        """Learn patterns from user interactions for future predictions"""
        
        # Learn user preferences and common query patterns
        if turn.query_type in [QueryType.PREDICTIVE_ANALYSIS, QueryType.STRATEGIC_THINKING]:
            
            # Track what types of predictions users ask for
            prediction_patterns = memory.learned_patterns.get('prediction_types', [])
            prediction_patterns.append(turn.query_type.value)
            memory.learned_patterns['prediction_types'] = prediction_patterns[-10:]  # Keep last 10
            
            # Track entities mentioned in predictive queries
            entity_patterns = memory.learned_patterns.get('entities_in_predictions', {})
            for entity in turn.entities_mentioned:
                entity_patterns[entity] = entity_patterns.get(entity, 0) + 1
            memory.learned_patterns['entities_in_predictions'] = entity_patterns
            
            # Learn from successful predictions
            if turn.predictive_insights and turn.predictive_insights.confidence_level > 0.7:
                successful_patterns = memory.learned_patterns.get('successful_prediction_patterns', [])
                successful_patterns.append({
                    'query_pattern': turn.user_query[:50],
                    'prediction_type': turn.predictive_insights.prediction_type.value,
                    'confidence': turn.predictive_insights.confidence_level
                })
                memory.learned_patterns['successful_prediction_patterns'] = successful_patterns[-5:]
    
    def _handle_followup_question(self, resolved_query: str, original_query: str, db_path: str, 
                                memory: ConversationMemory, domain_context: List[str], 
                                conversation_context: List[str]) -> Dict[str, Any]:
        """Handle follow-up questions using conversation context"""
        
        if not memory.turns or memory.last_query_results is None:
            return {
                "type": "followup_question",
                "response": "I don't have previous query results to expand on. Please ask a specific question about your property data.",
                "success": False
            }
        
        last_turn = memory.turns[-1]
        last_results = memory.last_query_results
        
        # Generate follow-up SQL based on previous query and current request
        followup_prompt = f"""
        Previous query: {last_turn.user_query}
        Previous SQL: {last_turn.sql_generated}
        Previous results summary: {len(last_results)} rows returned
        Previous results columns: {list(last_results.columns) if not last_results.empty else 'No results'}
        
        Current follow-up question: {original_query}
        Resolved query: {resolved_query}
        
        Context: {' '.join(domain_context[:2])}
        
        Generate a SQL query that answers the follow-up question based on the previous context.
        If the previous query was a count/aggregate, now show the detailed records.
        If it was a list, now show more details or filter further.
        
        Return only valid SQLite SQL.
        """
        
        try:
            sql_response = self.model.generate_content(followup_prompt)
            sql = self._clean_sql(sql_response.text)
            
            # Execute query
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            # Generate contextual response
            response_prompt = f"""
            This is a follow-up to: "{last_turn.user_query}"
            Current question: "{original_query}"
            
            Previous results: {len(last_results)} records
            Current results: {len(df)} records
            Current data: {df.head(3).to_dict() if not df.empty else 'No results'}
            
            Provide a conversational response that connects this to the previous query.
            Explain how this expands on or relates to the previous information.
            """
            
            response_gen = self.model.generate_content(response_prompt)
            response_text = response_gen.text
            
            # Generate follow-up suggestions
            suggestions = self._generate_followup_suggestions(df, original_query)
            
            return {
                "type": "followup_question",
                "sql": sql,
                "results": df,
                "response": response_text,
                "follow_up_suggestions": suggestions,
                "success": True,
                "context_connection": f"Following up on: {last_turn.user_query}"
            }
            
        except Exception as e:
            return {
                "type": "followup_question",
                "error": str(e),
                "response": f"I had trouble processing your follow-up question: {str(e)}",
                "success": False
            }
    
    def _handle_sql_query(self, query: str, db_path: str, memory: ConversationMemory,
                         domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Enhanced SQL query handling with memory context"""
        
        # Include conversation context in SQL generation
        context_info = ""
        if conversation_context:
            context_info = f"Conversation context: {' '.join(conversation_context[:2])}"
        
        sql_prompt = f"""
        {context_info}
        Domain knowledge: {' '.join(domain_context[:2])}
        
        Generate SQL for: {query}
        
        Consider any active filters or context from the conversation.
        Return only valid SQLite SQL.
        """
        
        try:
            sql_response = self.model.generate_content(sql_prompt)
            sql = self._clean_sql(sql_response.text)
            
            # Execute query
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            # Generate insights with conversation context
            insights = self._generate_contextual_insights(query, sql, df, memory, domain_context)
            
            # Generate follow-up suggestions
            suggestions = self._generate_followup_suggestions(df, query)
            
            return {
                "type": "sql_query",
                "sql": sql,
                "results": df,
                "insights": insights,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "sql_query",
                "sql": sql if 'sql' in locals() else "Error generating SQL",
                "error": str(e),
                "success": False
            }
    
    def _handle_trend_analysis(self, query: str, db_path: str, memory: ConversationMemory,
                             domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle trend analysis with enhanced analytics"""
        
        try:
            # Get time-series data for trend analysis
            trend_query = self._generate_trend_query(query)
            
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(trend_query, conn)
            conn.close()
            
            # Perform trend analysis
            trend_insights = self._analyze_trends(df, query)
            
            # Generate response
            response = self._generate_trend_response(query, trend_insights, df)
            
            suggestions = [
                "What factors are driving these trends?",
                "How do these trends compare to industry benchmarks?",
                "What should we do to improve these metrics?",
                "Can you predict future trends?"
            ]
            
            return {
                "type": "trend_analysis",
                "sql": trend_query,
                "results": df,
                "response": response,
                "trend_insights": trend_insights,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "trend_analysis",
                "error": str(e),
                "response": f"I encountered an error during trend analysis: {str(e)}",
                "success": False
            }
    
    def _handle_general_query(self, query: str, db_path: str, memory: ConversationMemory,
                            domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle general queries with context"""
        
        general_prompt = f"""
        Query: {query}
        Domain context: {' '.join(domain_context)}
        Conversation context: {' '.join(conversation_context[:2])}
        
        Provide a helpful response about property management.
        If this seems like it should involve data analysis, suggest a specific approach.
        If it involves predictions or strategic thinking, explain how to approach it.
        """
        
        try:
            response = self.model.generate_content(general_prompt)
            return {
                "type": "general_query",
                "response": response.text,
                "success": True
            }
        except Exception as e:
            return {
                "type": "general_query",
                "response": f"I encountered an error: {str(e)}",
                "success": False
            }
    
    def _generate_trend_query(self, query: str) -> str:
        """Generate SQL query for trend analysis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['payment', 'rent', 'income']):
            return """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as payment_count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount
            FROM payments 
            GROUP BY DATE(created_at)
            ORDER BY date
            """
        elif any(word in query_lower for word in ['maintenance', 'ticket']):
            return """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as ticket_count,
                priority,
                category
            FROM service_tickets 
            GROUP BY DATE(created_at), priority, category
            ORDER BY date
            """
        elif any(word in query_lower for word in ['occupancy', 'vacancy']):
            return """
            SELECT 
                DATE(created_at) as date,
                status,
                COUNT(*) as unit_count
            FROM units 
            GROUP BY DATE(created_at), status
            ORDER BY date
            """
        else:
            return """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as tenant_count
            FROM tenants 
            GROUP BY DATE(created_at)
            ORDER BY date
            """
    
    def _analyze_trends(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Analyze trends in the data"""
        insights = {
            "trend_direction": "unknown",
            "trend_strength": "weak",
            "key_observations": [],
            "recommendations": []
        }
        
        try:
            if df.empty:
                insights["key_observations"].append("No data available for trend analysis")
                return insights
            
            # Basic trend analysis
            if 'date' in df.columns and len(df) > 1:
                # Convert date column to datetime if it's not already
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Analyze numeric columns for trends
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if col != 'date' and len(df[col].dropna()) > 1:
                        # Simple trend calculation
                        values = df[col].dropna()
                        if len(values) >= 2:
                            start_val = values.iloc[0]
                            end_val = values.iloc[-1]
                            
                            if end_val > start_val * 1.1:
                                trend = "increasing"
                            elif end_val < start_val * 0.9:
                                trend = "decreasing"
                            else:
                                trend = "stable"
                            
                            insights["key_observations"].append(
                                f"{col.replace('_', ' ').title()} trend: {trend}"
                            )
                            
                            # Calculate percentage change
                            pct_change = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
                            if abs(pct_change) > 20:
                                insights["trend_strength"] = "strong"
                                insights["key_observations"].append(
                                    f"{col.replace('_', ' ').title()} changed by {pct_change:.1f}%"
                                )
            
            # Generate recommendations based on trends
            if "decreasing" in str(insights["key_observations"]):
                insights["recommendations"].append("Investigate factors causing declining metrics")
            if "increasing" in str(insights["key_observations"]):
                insights["recommendations"].append("Identify what's driving positive trends to replicate")
            
        except Exception as e:
            insights["error"] = str(e)
        
        return insights
    
    def _generate_trend_response(self, query: str, trend_insights: Dict[str, Any], df: pd.DataFrame) -> str:
        """Generate a comprehensive trend analysis response"""
        
        response = f"""## 📈 Trend Analysis

**Analysis Period:** {len(df)} data points analyzed

### Key Findings

**Overall Trend Strength:** {trend_insights.get('trend_strength', 'Unknown').title()}

**Key Observations:**
"""
        
        for observation in trend_insights.get('key_observations', []):
            response += f"- {observation}\n"
        
        if not trend_insights.get('key_observations'):
            response += "- Limited data available for comprehensive trend analysis\n"
        
        if trend_insights.get('recommendations'):
            response += "\n### 💡 Recommendations:\n"
            for rec in trend_insights['recommendations']:
                response += f"- {rec}\n"
        
        response += f"""
### 📊 Data Summary
- **Records Analyzed:** {len(df)}
- **Date Range:** {df['date'].min()} to {df['date'].max() if 'date' in df.columns and not df.empty else 'N/A'}

*Note: This analysis is based on available historical data. For more accurate trends, consider expanding the data collection period and including external market factors.*
"""
        
        return response
    
    def _generate_contextual_insights(self, query: str, sql: str, df: pd.DataFrame, 
                                    memory: ConversationMemory, domain_context: List[str]) -> str:
        """Generate humanized, formal insights that directly answer the query"""
        
        if df is None or df.empty:
            return "I wasn't able to find any records matching your criteria. You may want to check if the data exists or try adjusting your search parameters."
        
        # For COUNT queries - get the actual count value
        if len(df) == 1 and len(df.columns) == 1:
            count_value = df.iloc[0, 0]
            column_name = df.columns[0].lower()
            
            if 'count' in column_name:
                query_lower = query.lower()
                
                if 'tenant' in query_lower:
                    if count_value == 0:
                        return "Currently, there are no tenants in your database. This could indicate either an empty property portfolio or that tenant data hasn't been entered yet."
                    elif count_value == 1:
                        return "You currently have **1 tenant** in your property management system."
                    else:
                        return f"Based on your database records, you currently have **{count_value} tenants** under management."
                
                elif 'property' in query_lower or 'properties' in query_lower:
                    if count_value == 0:
                        return "No properties are currently recorded in your system."
                    elif count_value == 1:
                        return "You have **1 property** in your portfolio."
                    else:
                        return f"Your property portfolio consists of **{count_value} properties**."
                
                elif 'unit' in query_lower:
                    if count_value == 0:
                        return "No units are currently recorded in your system."
                    elif count_value == 1:
                        return "You have **1 unit** available in your property management system."
                    else:
                        return f"Your property portfolio includes **{count_value} units** across all properties."
                
                elif 'payment' in query_lower:
                    if count_value == 0:
                        return "No payment records were found matching your criteria."
                    elif count_value == 1:
                        return "I found **1 payment record** that matches your query."
                    else:
                        return f"There are **{count_value} payment records** that match your search criteria."
                
                elif 'maintenance' in query_lower or 'ticket' in query_lower:
                    if count_value == 0:
                        return "Great news! There are currently no maintenance tickets in your system."
                    elif count_value == 1:
                        return "You have **1 maintenance ticket** that requires attention."
                    else:
                        return f"There are currently **{count_value} maintenance tickets** in your system that may require attention."
                
                elif 'lease' in query_lower:
                    if count_value == 0:
                        return "No lease agreements were found matching your criteria."
                    elif count_value == 1:
                        return "I found **1 lease agreement** that matches your query."
                    else:
                        return f"There are **{count_value} lease agreements** that match your search criteria."
                
                elif 'overdue' in query_lower or 'late' in query_lower:
                    if count_value == 0:
                        return "Excellent! You have no overdue payments at this time. All tenants appear to be current with their rent."
                    elif count_value == 1:
                        return "You have **1 overdue payment** that requires follow-up."
                    else:
                        return f"There are **{count_value} overdue payments** that need your immediate attention."
                
                elif 'vacant' in query_lower:
                    if count_value == 0:
                        return "Wonderful! All your units are currently occupied. You have achieved 100% occupancy."
                    elif count_value == 1:
                        return "You have **1 vacant unit** that is available for new tenants."
                    else:
                        return f"You currently have **{count_value} vacant units** available for lease."
                
                else:
                    if count_value == 0:
                        return "No records were found matching your search criteria."
                    elif count_value == 1:
                        return "I found **1 record** that matches your query."
                    else:
                        return f"I found **{count_value} records** that match your search criteria."
        
        # For detailed queries (multiple rows/columns) - provide context-aware responses
        insights = []
        query_lower = query.lower()
        
        # Provide meaningful context based on the type of data returned
        if 'first_name' in df.columns or 'last_name' in df.columns or 'email' in df.columns:
            if len(df) == 1:
                insights.append("Here are the details for the tenant you requested:")
            else:
                insights.append(f"I've retrieved information for **{len(df)} tenants** as requested:")
        
        elif 'property_name' in df.columns or 'address' in df.columns:
            if len(df) == 1:
                insights.append("Here are the property details you requested:")
            else:
                insights.append(f"I've found **{len(df)} properties** matching your criteria:")
        
        elif 'unit_number' in df.columns:
            if len(df) == 1:
                insights.append("Here are the unit details:")
            else:
                insights.append(f"I've located **{len(df)} units** that match your search:")
        
        elif 'amount' in df.columns and 'due_date' in df.columns:
            if 'overdue' in query_lower:
                if len(df) == 1:
                    insights.append("Here is the overdue payment that requires attention:")
                else:
                    insights.append(f"I've identified **{len(df)} overdue payments** that need immediate follow-up:")
            else:
                if len(df) == 1:
                    insights.append("Here is the payment record you requested:")
                else:
                    insights.append(f"I've found **{len(df)} payment records** matching your criteria:")
        
        elif 'description' in df.columns and 'status' in df.columns:
            if len(df) == 1:
                insights.append("Here is the maintenance ticket information:")
            else:
                insights.append(f"I've found **{len(df)} maintenance tickets** in your system:")
        
        else:
            if len(df) == 1:
                insights.append("Here is the information you requested:")
            else:
                insights.append(f"I've found **{len(df)} records** that match your query:")
        
        # Add specific insights based on data content
        if 'overdue' in query_lower and 'amount' in df.columns:
            total_overdue = df['amount'].sum()
            insights.append(f"The total amount overdue is **${total_overdue:,.2f}**.")
        
        elif 'vacant' in query_lower and len(df) > 0:
            insights.append("These units are currently available for new tenant applications.")
        
        elif 'maintenance' in query_lower and 'priority' in df.columns:
            emergency_count = len(df[df['priority'].str.lower() == 'emergency']) if 'priority' in df.columns else 0
            if emergency_count > 0:
                insights.append(f"**Important:** {emergency_count} of these tickets are marked as emergency priority.")
        
        elif 'payment' in query_lower and 'amount' in df.columns and len(df) > 1:
            total_amount = df['amount'].sum()
            avg_amount = df['amount'].mean()
            insights.append(f"Total amount: **${total_amount:,.2f}** | Average: **${avg_amount:,.2f}**")
        
        # Return the formatted response
        if insights:
            return "\n\n".join(insights)
        else:
            return f"I've successfully retrieved **{len(df)} records** based on your request."
    
    def _generate_followup_suggestions(self, df: pd.DataFrame, query: str) -> List[str]:
        """Generate intelligent follow-up suggestions based on results"""
        
        if df is None or df.empty:
            return ["Try a different search criteria", "Check if the data exists in the database"]
        
        suggestions = []
        columns = df.columns.tolist()
        
        # Enhanced suggestions based on result type and columns
        if 'first_name' in columns or 'last_name' in columns:
            suggestions.extend([
                "Show me their contact information",
                "What are their lease details?",
                "Predict tenant churn risk for these tenants"
            ])
        
        if 'property_name' in columns or 'address' in columns:
            suggestions.extend([
                "How many units are in these properties?",
                "What's the occupancy forecast?",
                "Analyze the strategic value of these properties"
            ])
        
        if 'amount' in columns or 'rent_amount' in columns:
            suggestions.extend([
                "Forecast future cash flow trends",
                "Which payments are at risk?",
                "What's the predicted collection rate?"
            ])
        
        if len(df) > 10:
            suggestions.append("Show me just the top 5 results")
        
        # Query-specific suggestions with predictive elements
        if 'count' in query.lower():
            suggestions.extend([
                "Show me the detailed list",
                "What trends do you see in this data?"
            ])
        elif 'overdue' in query.lower():
            suggestions.extend([
                "Predict which tenants are at risk of late payment",
                "What strategies can reduce payment delays?"
            ])
        elif 'vacant' in query.lower():
            suggestions.extend([
                "Forecast occupancy rates for next quarter",
                "What's our competitive position for these units?"
            ])
        elif 'maintenance' in query.lower():
            suggestions.extend([
                "Predict future maintenance needs",
                "Which categories need preventive attention?"
            ])
        
        # Add strategic thinking suggestions
        suggestions.extend([
            "What strategic insights can you provide?",
            "How does this compare to industry benchmarks?"
        ])
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _clean_sql(self, raw_sql: str) -> str:
        """Clean SQL by removing markdown formatting"""
        sql = re.sub(r"```[^\n]*\n", "", raw_sql)
        sql = re.sub(r"\n```", "", sql)
        return sql.strip()
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get a summary of the conversation"""
        if session_id not in self.memory_store:
            return "No conversation history"
        
        memory = self.memory_store[session_id]
        if not memory.turns:
            return "No questions asked yet"
        
        summary_parts = []
        for turn in memory.turns[-5:]:  # Last 5 turns
            summary_parts.append(f"Q: {turn.user_query[:50]}...")
        
        return " | ".join(summary_parts)

# Enhanced Streamlit UI with Predictive Analytics Features
def enhanced_sidebar_with_sessions():
    """Enhanced sidebar with session management and analytics features"""
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Database upload
        db_file = st.file_uploader("Upload SQLite Database", type=["db", "sqlite"])
        if db_file:
            db_path = "/tmp/uploaded.db"
            with open(db_path, "wb") as f:
                f.write(db_file.getbuffer())
        else:
            db_path = "database.db"
        
        st.divider()
        
        # Analytics Settings
        st.header("📊 Analytics Settings")
        
        analytics_mode = st.selectbox(
            "Default Analysis Mode",
            ["Basic", "Predictive", "Strategic", "Comprehensive"],
            index=2,
            help="Choose the default level of analysis for your queries"
        )
        
        confidence_threshold = st.slider(
            "Prediction Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Minimum confidence level for displaying predictions"
        )
        
        show_thinking = st.checkbox(
            "Show AI Reasoning Process",
            value=True,
            help="Display the AI's thinking process and methodology"
        )
        
        st.session_state.analytics_settings = {
            "mode": analytics_mode,
            "confidence_threshold": confidence_threshold,
            "show_thinking": show_thinking
        }
        
        st.divider()
        
        # Session Management
        st.header("📚 Session Management")
        
        # Load sessions on startup
        if 'sessions_loaded' not in st.session_state:
            load_sessions_from_disk(st.session_state.agent)
            st.session_state.sessions_loaded = True
        
        # Current session info
        current_memory = st.session_state.agent.get_or_create_memory(st.session_state.session_id)
        current_summary = get_session_summary(current_memory)
        
        st.info(f"""
        **Current Session:** `{st.session_state.session_id}`  
        **Questions Asked:** {current_summary['question_count']}  
        **Started:** {current_summary['last_activity'] if current_summary['question_count'] > 0 else 'Just now'}
        """)
        
        # Session actions
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🆕 New Session", use_container_width=True):
                # Save current sessions before creating new one
                save_sessions_to_disk(st.session_state.agent)
                
                # Create new session
                st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
                st.session_state.conversation_history = []
                st.success("New session created!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save Sessions", use_container_width=True):
                save_sessions_to_disk(st.session_state.agent)
                st.success("Sessions saved!")
        
        # Previous Sessions (same as before but condensed for space)
        if len(st.session_state.agent.memory_store) > 1:
            st.subheader("📋 Previous Sessions")
            
            sessions = []
            for session_id, memory in st.session_state.agent.memory_store.items():
                if session_id != st.session_state.session_id and memory.turns:
                    summary = get_session_summary(memory)
                    sessions.append((session_id, memory, summary))
            
            sessions.sort(key=lambda x: x[2]['last_activity'], reverse=True)
            
            for session_id, memory, summary in sessions[:3]:  # Show last 3 sessions
                with st.expander(f"📄 {summary['title'][:25]}...", expanded=False):
                    st.write(f"**Questions:** {summary['question_count']}")
                    st.write(f"**Last:** {summary['last_activity']}")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"🔄 Switch", key=f"switch_{session_id}", use_container_width=True):
                            if st.session_state.session_id in st.session_state.agent.memory_store:
                                current_mem = st.session_state.agent.memory_store[st.session_state.session_id]
                                st.session_state.conversation_history = [
                                    (turn.user_query, turn.ai_response, turn.timestamp)
                                    for turn in current_mem.turns
                                ]
                            
                            st.session_state.session_id = session_id
                            
                            selected_memory = st.session_state.agent.memory_store[session_id]
                            st.session_state.conversation_history = [
                                (turn.user_query, turn.ai_response, turn.timestamp)
                                for turn in selected_memory.turns
                            ]
                            
                            st.success(f"Switched to session {session_id}")
                            st.rerun()
                    
                    with col_b:
                        if st.button(f"🗑️ Delete", key=f"delete_{session_id}", use_container_width=True):
                            del st.session_state.agent.memory_store[session_id]
                            save_sessions_to_disk(st.session_state.agent)
                            st.success("Deleted!")
                            st.rerun()
        
        return db_path

# Main function with enhanced predictive features
def main():
    st.set_page_config(page_title="🏠 AI Property Management Assistant", layout="wide")
    
    st.title("🏠 AI Property Management Assistant")
    st.markdown("*Powered by Predictive Analytics, Strategic AI, and Conversational Memory*")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = PropertyRAGSystem()
    
    if 'agent' not in st.session_state:
        st.session_state.agent = PropertyManagementAgent(st.session_state.rag_system)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'analytics_settings' not in st.session_state:
        st.session_state.analytics_settings = {
            "mode": "Strategic",
            "confidence_threshold": 0.6,
            "show_thinking": True
        }
    
    # Enhanced sidebar with predictive analytics settings
    db_path = enhanced_sidebar_with_sessions()
    
    # Auto-save sessions periodically
    if len(st.session_state.agent.memory_store) > 0:
        save_sessions_to_disk(st.session_state.agent)
    
    # Main interface
    st.header("💬 Conversation")
    
    # Display current session info with analytics capabilities
    current_memory = st.session_state.agent.get_or_create_memory(st.session_state.session_id)
    if current_memory.turns:
        analytics_info = f" | Analytics: {st.session_state.analytics_settings['mode']} Mode"
        st.caption(f"Session: {st.session_state.session_id} | Questions: {len(current_memory.turns)} | Started: {current_memory.turns[0].timestamp.strftime('%Y-%m-%d %H:%M')}{analytics_info}")
    else:
        st.caption(f"Session: {st.session_state.session_id} | New session | Analytics: {st.session_state.analytics_settings['mode']} Mode")
    
    # Display conversation history
    if st.session_state.conversation_history:
        with st.expander("📜 Conversation History", expanded=False):
            for i, (user_msg, ai_response, timestamp) in enumerate(st.session_state.conversation_history):
                st.write(f"**{timestamp.strftime('%H:%M:%S')} - You:** {user_msg}")
                st.write(f"**AI:** {ai_response[:200]}...")
                if i < len(st.session_state.conversation_history) - 1:
                    st.divider()
    
    # Enhanced query input with examples
    st.subheader("🎯 Ask Your Question")
    
    # Example queries with predictive/strategic focus
    example_queries = [
        "How many tenants do we have?",
        "Predict our occupancy rate for next quarter",
        "What's our strategic position in the market?",
        "Analyze cash flow trends and forecast risks",
        "Show me maintenance tickets and predict future needs",
        "What scenarios should we plan for?",
        "Which units are vacant and what's the forecast?",
        "Strategic recommendations for portfolio growth"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_value = ""
        if 'suggested_query' in st.session_state:
            default_value = st.session_state.suggested_query
            del st.session_state.suggested_query
        
        query = st.text_area(
            "Your question:",
            placeholder="e.g., 'Predict our cash flow for next 6 months' or 'What strategic opportunities do we have?'",
            height=100,
            value=default_value,
            key=f"current_query_{st.session_state.session_id}"
        )
    
    with col2:
        st.write("**Example Queries:**")
        for i, example in enumerate(example_queries[:4]):
            if st.button(f"💡 {example[:30]}...", key=f"example_{i}", use_container_width=True):
                st.session_state.suggested_query = example
                st.rerun()
    
    # Process query
    if st.button("🚀 Ask Question", type="primary", use_container_width=True) and query:
        with st.spinner("🤔 Processing your question with advanced analytics..."):
            result = st.session_state.agent.process_query(
                query, 
                db_path, 
                st.session_state.session_id
            )
        
        # Add to conversation history
        timestamp = datetime.now()
        ai_response = result.get('response', result.get('insights', 'Processed successfully'))
        st.session_state.conversation_history.append((query, ai_response, timestamp))
        
        # Display results based on type with enhanced predictive features
        if result["success"]:
            
            # Show context awareness
            if result.get("context_connection"):
                st.info(f"🔗 {result['context_connection']}")
            
            # Enhanced display for predictive analysis
            if result["type"] == "predictive_analysis":
                st.subheader("🔮 Predictive Analysis Results")
                
                # Show analysis type and confidence
                analysis_type = result.get("analysis_type", "Predictive Analysis")
                if result.get("predictive_insights"):
                    confidence = result["predictive_insights"].confidence_level
                    confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                    st.success(f"**{analysis_type}** | Confidence: {confidence_color} {confidence:.1%}")
                
                # Show thinking process if enabled
                if st.session_state.analytics_settings["show_thinking"] and result.get("thinking_process"):
                    with st.expander("🧠 AI Reasoning Process", expanded=False):
                        thinking = result["thinking_process"]
                        st.write("**Problem Analysis:**", thinking.problem_analysis)
                        st.write("**Data Assessment:**", thinking.data_assessment)
                        st.write("**Methodology:**", thinking.methodology)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Key Assumptions:**")
                            for assumption in thinking.assumptions:
                                st.write(f"• {assumption}")
                        with col_b:
                            st.write("**Limitations:**")
                            for limitation in thinking.limitations:
                                st.write(f"• {limitation}")
                
                # Show SQL if generated
                if result.get("sql"):
                    with st.expander("🔍 Data Query Used"):
                        st.code(result["sql"], language="sql")
                
                # Show results data
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("📊 Underlying Data")
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Enhanced metrics for predictions
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Data Points", len(result["results"]))
                    with col2:
                        st.metric("Variables", len(result["results"].columns))
                    with col3:
                        if result.get("predictive_insights"):
                            st.metric("Confidence", f"{result['predictive_insights'].confidence_level:.1%}")
                    with col4:
                        if result.get("predictive_insights"):
                            st.metric("Time Horizon", result["predictive_insights"].time_horizon)
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Analysis Data",
                        csv,
                        f"predictive_analysis_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show predictive insights
                if result.get("predictive_insights"):
                    insights = result["predictive_insights"]
                    
                    st.subheader("🎯 Key Predictions")
                    # Display key metrics in a nice format
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        for key, value in insights.key_metrics.items():
                            if key not in ['error', 'confidence', 'recommendations'] and isinstance(value, (int, float)):
                                st.metric(key.replace('_', ' ').title(), f"{value:,.2f}")
                    
                    with metrics_col2:
                        if insights.recommendations:
                            st.write("**Recommended Actions:**")
                            for rec in insights.recommendations:
                                st.write(f"✅ {rec}")
                    
                    # Risk factors if available
                    if insights.risk_factors:
                        st.warning("**Risk Factors to Monitor:**")
                        for risk in insights.risk_factors:
                            st.write(f"⚠️ {risk}")
                
                # Show main response
                st.subheader("💡 Analysis Summary")
                st.markdown(result["response"])
            
            # Enhanced display for strategic thinking
            elif result["type"] == "strategic_thinking":
                st.subheader("🎯 Strategic Analysis")
                
                # Show thinking process if enabled
                if st.session_state.analytics_settings["show_thinking"] and result.get("thinking_process"):
                    with st.expander("🧠 Strategic Reasoning", expanded=False):
                        thinking = result["thinking_process"]
                        st.write("**Strategic Framework:**", thinking.methodology)
                        st.write("**Analysis Scope:**", thinking.data_assessment)
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.write("**Strategic Assumptions:**")
                            for assumption in thinking.assumptions:
                                st.write(f"• {assumption}")
                        with col_b:
                            st.write("**Alternative Approaches:**")
                            for alternative in thinking.alternative_approaches:
                                st.write(f"• {alternative}")
                
                # Show strategic analysis results
                if result.get("strategic_analysis"):
                    analysis = result["strategic_analysis"]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Portfolio Strength", analysis.get('portfolio_strength', 'Unknown').title())
                    with col2:
                        advantages_count = len(analysis.get('competitive_advantages', []))
                        st.metric("Competitive Advantages", advantages_count)
                    with col3:
                        opportunities_count = len(analysis.get('growth_opportunities', []))
                        st.metric("Growth Opportunities", opportunities_count)
                
                # Show results if any
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("📊 Strategic Data")
                    st.dataframe(result["results"], use_container_width=True)
                
                # Show main response
                st.markdown(result["response"])
            
            # Enhanced display for scenario planning
            elif result["type"] == "scenario_planning":
                st.subheader("🎲 Scenario Planning")
                
                # Show thinking process if enabled
                if st.session_state.analytics_settings["show_thinking"] and result.get("thinking_process"):
                    with st.expander("🧠 Scenario Methodology", expanded=False):
                        thinking = result["thinking_process"]
                        st.write("**Planning Framework:**", thinking.methodology)
                        st.write("**Scenario Assumptions:**")
                        for assumption in thinking.assumptions:
                            st.write(f"• {assumption}")
                
                # Show scenario results in tabs
                if result.get("scenario_analysis") and result["scenario_analysis"].get("scenarios"):
                    scenarios = result["scenario_analysis"]["scenarios"]
                    
                    tab1, tab2, tab3 = st.tabs(["🌟 Optimistic", "📊 Baseline", "⚠️ Pessimistic"])
                    
                    with tab1:
                        if "optimistic" in scenarios:
                            opt = scenarios["optimistic"]
                            st.write(f"**Description:** {opt.get('description', 'N/A')}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Occupancy Change", opt.get('occupancy_change', 'N/A'))
                            with col2:
                                st.metric("Rent Change", opt.get('rent_change', 'N/A'))
                            with col3:
                                st.metric("Maintenance Change", opt.get('maintenance_change', 'N/A'))
                            
                            st.write(f"**Strategy:** {opt.get('strategy', 'No strategy defined')}")
                    
                    with tab2:
                        if "baseline" in scenarios:
                            base = scenarios["baseline"]
                            st.write(f"**Description:** {base.get('description', 'N/A')}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Occupancy Change", base.get('occupancy_change', 'N/A'))
                            with col2:
                                st.metric("Rent Change", base.get('rent_change', 'N/A'))
                            with col3:
                                st.metric("Maintenance Change", base.get('maintenance_change', 'N/A'))
                            
                            st.write(f"**Strategy:** {base.get('strategy', 'No strategy defined')}")
                    
                    with tab3:
                        if "pessimistic" in scenarios:
                            pess = scenarios["pessimistic"]
                            st.write(f"**Description:** {pess.get('description', 'N/A')}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Occupancy Change", pess.get('occupancy_change', 'N/A'))
                            with col2:
                                st.metric("Rent Change", pess.get('rent_change', 'N/A'))
                            with col3:
                                st.metric("Maintenance Change", pess.get('maintenance_change', 'N/A'))
                            
                            st.write(f"**Strategy:** {pess.get('strategy', 'No strategy defined')}")
                
                # Show scenario data if available
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("📊 Scenario Comparison")
                    st.dataframe(result["results"], use_container_width=True)
                
                # Show main response
                st.markdown(result["response"])
            
            # Enhanced display for trend analysis
            elif result["type"] == "trend_analysis":
                st.subheader("📈 Trend Analysis")
                
                # Show SQL
                if result.get("sql"):
                    with st.expander("🔍 Trend Query"):
                        st.code(result["sql"], language="sql")
                
                # Show trend insights
                if result.get("trend_insights"):
                    insights = result["trend_insights"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trend Direction", insights.get('trend_direction', 'Unknown').title())
                    with col2:
                        st.metric("Trend Strength", insights.get('trend_strength', 'Unknown').title())
                    with col3:
                        observations_count = len(insights.get('key_observations', []))
                        st.metric("Key Observations", observations_count)
                
                # Show results
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("📊 Trend Data")
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Try to create a simple trend visualization
                    if 'date' in result["results"].columns:
                        try:
                            trend_df = result["results"].copy()
                            trend_df['date'] = pd.to_datetime(trend_df['date'])
                            
                            numeric_cols = trend_df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                st.subheader("📈 Trend Visualization")
                                chart_col = numeric_cols[0]
                                st.line_chart(trend_df.set_index('date')[chart_col])
                        except:
                            pass  # Skip visualization if it fails
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Trend Data",
                        csv,
                        f"trend_analysis_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show main response
                st.markdown(result["response"])
            
            # Standard follow-up question display
            elif result["type"] == "followup_question":
                st.subheader("🔄 Follow-up Response")
                
                # Show how the question was resolved
                memory = st.session_state.agent.get_or_create_memory(st.session_state.session_id)
                if len(memory.turns) >= 2:
                    prev_turn = memory.turns[-2]
                    st.success(f"💡 **Connected to previous query:** {prev_turn.user_query}")
                
                # Show SQL if generated
                if result.get("sql"):
                    with st.expander("🔍 Generated SQL"):
                        st.code(result["sql"], language="sql")
                
                # Show results
                if result.get("results") is not None and not result["results"].empty:
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        f"followup_results_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show response
                st.markdown("**AI Response:**")
                st.write(result["response"])
            
            # Standard SQL query display
            elif result["type"] == "sql_query":
                st.subheader("📊 Query Results")
                
                # Show SQL
                with st.expander("🔍 Generated SQL"):
                    st.code(result["sql"], language="sql")
                
                # Show results
                if result.get("results") is not None and not result["results"].empty:
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Quick stats
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Rows", len(result["results"]))
                    with col_b:
                        st.metric("Columns", len(result["results"].columns))
                    with col_c:
                        # For COUNT queries, show the actual count value
                        df = result["results"]
                        if len(df) == 1 and len(df.columns) == 1 and 'count' in df.columns[0].lower():
                            actual_count = df.iloc[0, 0]
                            st.metric("Count", actual_count)
                        elif df.select_dtypes(include=[np.number]).columns.any():
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                avg_val = df[numeric_cols[0]].mean()
                                st.metric(f"Avg {numeric_cols[0]}", f"{avg_val:.2f}")
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        f"query_results_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No results found.")
                
                # Show insights
                if result.get("insights"):
                    st.subheader("💡 Answer")
                    insights_text = result["insights"]
                    if insights_text and not insights_text.startswith("SELECT"):
                        st.markdown(insights_text)
            
            # General query display
            elif result["type"] == "general_query":
                st.subheader("💬 AI Response")
                st.markdown(result["response"])
            
            # Show follow-up suggestions for ALL successful results
            if result.get("follow_up_suggestions"):
                st.subheader("🎯 Suggested Follow-up Questions")
                
                # Enhanced suggestions with predictive/strategic focus
                suggestion_categories = {
                    "🔮 Predictive": [s for s in result["follow_up_suggestions"] if any(word in s.lower() for word in ['predict', 'forecast', 'future', 'trend'])],
                    "🎯 Strategic": [s for s in result["follow_up_suggestions"] if any(word in s.lower() for word in ['strategic', 'strategy', 'opportunity', 'risk', 'competitive'])],
                    "📊 Analytical": [s for s in result["follow_up_suggestions"] if s not in [s for s in result["follow_up_suggestions"] if any(word in s.lower() for word in ['predict', 'forecast', 'future', 'trend', 'strategic', 'strategy', 'opportunity', 'risk', 'competitive'])]]
                }
                
                # Display suggestions in categories
                cols = st.columns(len([cat for cat, sugs in suggestion_categories.items() if sugs]))
                col_idx = 0
                
                for category, suggestions in suggestion_categories.items():
                    if suggestions:
                        with cols[col_idx]:
                            st.write(f"**{category}**")
                            for i, suggestion in enumerate(suggestions[:2]):  # Limit to 2 per category
                                suggestion_key = f"suggestion_{st.session_state.session_id}_{category}_{i}_{len(st.session_state.conversation_history)}_{hash(suggestion)}"
                                
                                if st.button(f"🔹 {suggestion[:35]}...", key=suggestion_key, use_container_width=True):
                                    st.session_state.suggested_query = suggestion
                                    st.rerun()
                        col_idx += 1
                
                # If no categorized suggestions, show regular layout
                if not any(suggestion_categories.values()):
                    cols = st.columns(2)
                    for i, suggestion in enumerate(result["follow_up_suggestions"]):
                        suggestion_key = f"suggestion_{st.session_state.session_id}_{i}_{len(st.session_state.conversation_history)}_{hash(suggestion)}"
                        
                        with cols[i % 2]:
                            if st.button(f"🔹 {suggestion}", key=suggestion_key, use_container_width=True):
                                st.session_state.suggested_query = suggestion
                                st.rerun()
        
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
            if result.get("sql"):
                st.code(result["sql"], language="sql")
    
    # Quick access to common predictive queries
    st.divider()
    st.subheader("🚀 Quick Analytics")
    
    quick_cols = st.columns(4)
    
    quick_queries = [
        ("🔮 Occupancy Forecast", "Predict our occupancy rate for the next 6 months"),
        ("💰 Cash Flow Analysis", "Analyze cash flow trends and predict future revenue"),
        ("🏠 Strategic Position", "What's our strategic position and growth opportunities?"),
        ("🔧 Maintenance Prediction", "Predict future maintenance needs and costs")
    ]
    
    for i, (title, query_text) in enumerate(quick_queries):
        with quick_cols[i]:
            if st.button(title, key=f"quick_{i}", use_container_width=True):
                st.session_state.suggested_query = query_text
                st.rerun()

if __name__ == "__main__":
    main()
