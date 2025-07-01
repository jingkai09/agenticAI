# enhanced_property_agent_with_memory.py

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
import time

import streamlit as st
import pandas as pd
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    execution_time: float = 0.0
    confidence_score: float = 1.0

@dataclass
class ConversationMemory:
    """Maintains conversation context and memory"""
    session_id: str
    turns: List[ConversationTurn]
    current_context: Dict[str, Any]
    entity_references: Dict[str, Any]  # Maps pronouns/references to actual entities
    active_filters: Dict[str, Any]      # Current filters being applied
    last_query_results: Optional[pd.DataFrame]
    conversation_summary: str
    user_preferences: Dict[str, Any] = None

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
                # Find the most recent relevant entity
                resolved_entity = self._find_recent_entity(recent_turns, possible_entities)
                if resolved_entity:
                    resolved_query = resolved_query.replace(pronoun, resolved_entity)
        
        # Handle specific follow-up patterns
        if any(phrase in resolved_query for phrase in ['who are they', 'what are they', 'show me them']):
            if memory.last_query_results is not None and not memory.last_query_results.empty:
                # Determine what type of data was last queried
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
    """Enhanced RAG system with conversation memory"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        self.vector_store = None
        self.documents = []
        self.conversation_embeddings = {}  # Store embeddings of conversation turns
        self._initialize_domain_knowledge()
    
    def _initialize_domain_knowledge(self):
        """Initialize with enhanced property management domain knowledge"""
        domain_docs = [
            """Property Management Best Practices:
            - Regular property inspections should be conducted quarterly
            - Rent collection should be automated with late fees after 5 days
            - Maintenance requests should be categorized by urgency: Emergency (24hrs), Urgent (3 days), Standard (7 days)
            - Tenant screening should include credit check, employment verification, and references
            - Security deposits typically equal 1-2 months rent depending on local laws
            - Monthly rent-to-income ratio should not exceed 30% for qualified tenants""",
            
            """Common SQL Patterns for Property Management:
            - Count tenants: SELECT COUNT(*) FROM tenants WHERE status = 'active'
            - List tenant details: SELECT first_name, last_name, email, phone FROM tenants
            - Find overdue rent: SELECT t.first_name, t.last_name, p.amount, p.due_date FROM tenants t JOIN leases l ON t.id = l.tenant_id JOIN payments p ON l.id = p.lease_id WHERE p.due_date < date('now') AND p.paid_on IS NULL
            - Active leases: SELECT * FROM leases WHERE end_date > date('now') AND status = 'active'
            - Maintenance by priority: SELECT * FROM service_tickets ORDER BY CASE priority WHEN 'emergency' THEN 1 WHEN 'urgent' THEN 2 ELSE 3 END
            - Tenant payment history: SELECT tenant_id, COUNT(*) as payment_count, SUM(amount) as total_paid FROM payments GROUP BY tenant_id
            - Vacancy rate: SELECT (COUNT(CASE WHEN status = 'vacant' THEN 1 END) * 100.0 / COUNT(*)) as vacancy_rate FROM units""",
            
            """Follow-up Query Patterns:
            - After "how many tenants": User might ask "who are they", "show me their details", "what are their contact info"
            - After property count: User might ask "which properties", "show me the addresses", "what are the property details"
            - After maintenance tickets: User might ask "what type of issues", "who reported them", "when were they created"
            - After payment queries: User might ask "which tenants", "what amounts", "when are they due"
            - After lease queries: User might ask "show lease details", "what are the terms", "when do they expire"
            - Common follow-ups: "show me more details", "who are they", "what about X", "tell me more", "expand on that" """,
            
            """Context Resolution Examples:
            - "how many tenants" → "who are they" = SELECT first_name, last_name, email, phone FROM tenants
            - "overdue payments" → "show them" = SELECT tenant details with overdue payment information
            - "maintenance tickets" → "what type" = SELECT category, subcategory, description FROM service_tickets
            - "vacant units" → "which ones" = SELECT unit_number, property_name, floor FROM units WHERE status = 'vacant'
            - "expensive properties" → "show addresses" = SELECT name, address_line1, city FROM properties ORDER BY (some expense metric)"""
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
            if turn.timestamp not in [t.timestamp for t in memory.turns[-1:]]:  # Don't include current turn
                turn_context = f"Previous: {turn.user_query} -> {turn.ai_response[:200]}..."
                conversation_context.append(turn_context)
        
        return domain_context, conversation_context

class PropertyManagementAgent:
    """Enhanced agentic AI system with comprehensive memory and analytics"""
    
    def __init__(self, rag_system: PropertyRAGSystem):
        self.rag_system = rag_system
        self.entity_extractor = EntityExtractor()
        self.context_resolver = ContextResolver()
        self.memory_store = {}  # Session ID -> ConversationMemory
        self.analytics = {
            'total_queries': 0,
            'successful_queries': 0,
            'query_types': {},
            'avg_response_time': 0,
            'user_satisfaction': []
        }
        
        self.model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        return """
        You are an expert Property Management AI Agent with advanced memory and context awareness.
        
        CAPABILITIES:
        - Generate and execute SQL queries for property management databases
        - Maintain conversation context and memory across multiple turns
        - Resolve pronouns and references from previous conversations
        - Provide follow-up answers based on previous queries
        - Generate business insights and recommendations
        - Create data visualizations and analytics
        
        DATABASE SCHEMA:
        - tenants(id, timestamp, first_name, last_name, email, phone, date_of_birth, created_at)
        - properties(id, timestamp, name, address_line1, address_line2, city, state, postal_code, country, created_at)
        - units(id, timestamp, property_id, unit_number, floor, bedrooms, bathrooms, square_feet, status, created_at)
        - leases(id, timestamp, tenant_id, unit_id, start_date, end_date, rent_amount, security_deposit, status, created_at)
        - payments(id, timestamp, lease_id, payment_type, billing_period, due_date, amount, method, paid_on, reference_number, created_at)
        - service_tickets(id, timestamp, lease_id, raised_by, assigned_to, category, subcategory, description, status, priority, created_at, updated_at)
        
        MEMORY RULES:
        1. Always consider conversation context when interpreting queries
        2. Resolve pronouns (they, them, it, those) based on previous queries
        3. For follow-up questions, reference previous results appropriately
        4. Maintain entity references across conversation turns
        5. Suggest logical follow-up questions based on current results
        
        FOLLOW-UP PATTERNS:
        - "how many X" → "who are they" = show detailed information about X
        - "list X" → "show me more details" = expand with additional columns
        - "find overdue Y" → "what amounts" = show specific amounts and dates
        - Count queries → Detail queries = expand from aggregate to individual records
        
        RESPONSE FORMAT:
        - Always provide clear, contextual responses
        - Include SQL when executing database queries
        - Offer follow-up suggestions when appropriate
        - Explain how you're using previous context when relevant
        - Return only valid SQLite SQL (no markdown formatting)
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
                user_preferences={}
            )
        return self.memory_store[session_id]
    
    def process_query(self, user_query: str, db_path: str, session_id: str = "default") -> Dict[str, Any]:
        """Main processing pipeline with memory integration and analytics"""
        
        start_time = time.time()
        self.analytics['total_queries'] += 1
        
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
        
        # Classify query type
        query_type = self._classify_intent(resolved_query, memory)
        
        # Update analytics
        if query_type.value not in self.analytics['query_types']:
            self.analytics['query_types'][query_type.value] = 0
        self.analytics['query_types'][query_type.value] += 1
        
        # Process based on type
        try:
            if query_type == QueryType.FOLLOWUP_QUESTION:
                result = self._handle_followup_question(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
            elif query_type == QueryType.SQL_QUERY:
                result = self._handle_sql_query(resolved_query, db_path, memory, domain_context, conversation_context)
            elif query_type == QueryType.TREND_ANALYSIS:
                result = self._handle_trend_analysis(resolved_query, db_path, memory, domain_context, conversation_context)
            else:
                result = self._handle_general_query(resolved_query, db_path, memory, domain_context, conversation_context)
            
            if result.get('success', False):
                self.analytics['successful_queries'] += 1
                
        except Exception as e:
            result = {
                "type": "error",
                "error": str(e),
                "response": f"An error occurred: {str(e)}",
                "success": False
            }
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self.analytics['avg_response_time'] = (self.analytics['avg_response_time'] * (self.analytics['total_queries'] - 1) + execution_time) / self.analytics['total_queries']
        
        # Create conversation turn
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
            execution_time=execution_time,
            confidence_score=result.get('confidence_score', 1.0)
        )
        
        # Update memory
        memory.turns.append(turn)
        if 'results' in result and isinstance(result['results'], pd.DataFrame):
            memory.last_query_results = result['results']
        
        # Add to RAG system
        self.rag_system.add_conversation_turn(turn)
        
        # Add execution metrics to result
        result['execution_time'] = execution_time
        result['confidence_score'] = turn.confidence_score
        
        return result
    
    def _classify_intent(self, query: str, memory: ConversationMemory) -> QueryType:
        """Enhanced intent classification considering conversation context"""
        
        # Check for obvious follow-up patterns
        followup_indicators = [
            'who are they', 'what are they', 'show me them', 'show them', 
            'tell me more', 'more details', 'expand on that', 'show me more',
            'which ones', 'what about', 'how about', 'and them', 'those too'
        ]
        
        if any(indicator in query.lower() for indicator in followup_indicators):
            return QueryType.FOLLOWUP_QUESTION
        
        # Check if this relates to previous query results
        if memory.last_query_results is not None and len(memory.turns) > 0:
            last_turn = memory.turns[-1]
            
            # Common follow-up patterns after count queries
            if 'count' in last_turn.sql_generated.lower() if last_turn.sql_generated else False:
                detail_words = ['details', 'names', 'list', 'show', 'who', 'which', 'what']
                if any(word in query.lower() for word in detail_words):
                    return QueryType.FOLLOWUP_QUESTION
        
        # Check for trend analysis keywords
        trend_keywords = ['trend', 'over time', 'monthly', 'yearly', 'pattern', 'analytics', 'dashboard']
        if any(keyword in query.lower() for keyword in trend_keywords):
            return QueryType.TREND_ANALYSIS
        
        # Standard classification
        classification_prompt = f"""
        Classify this property management query:
        
        Query: {query}
        Recent context: {memory.turns[-1].user_query if memory.turns else 'None'}
        
        Categories:
        1. sql_query - Direct data queries
        2. trend_analysis - Pattern analysis, trends
        3. followup_question - Follow-up to previous query
        4. general_query - General questions
        
        Return only the category name.
        """
        
        try:
            response = self.model.generate_content(classification_prompt)
            intent = response.text.strip().lower()
            return QueryType(intent) if intent in [t.value for t in QueryType] else QueryType.SQL_QUERY
        except:
            return QueryType.SQL_QUERY
    
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
                "context_connection": f"Following up on: {last_turn.user_query}",
                "confidence_score": 0.9
            }
            
        except Exception as e:
            return {
                "type": "followup_question",
                "error": str(e),
                "response": f"I had trouble processing your follow-up question: {str(e)}",
                "success": False,
                "confidence_score": 0.1
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
            
            # Create visualization if applicable
            visualization = self._create_visualization(df, query)
            
            result = {
                "type": "sql_query",
                "sql": sql,
                "results": df,
                "insights": insights,
                "follow_up_suggestions": suggestions,
                "success": True,
                "confidence_score": 0.95
            }
            
            if visualization:
                result["visualization"] = visualization
            
            return result
            
        except Exception as e:
            return {
                "type": "sql_query",
                "sql": sql if 'sql' in locals() else "Error generating SQL",
                "error": str(e),
                "success": False,
                "confidence_score": 0.1
            }
    
    def _create_visualization(self, df: pd.DataFrame, query: str) -> Optional[Dict[str, Any]]:
        """Create appropriate visualization for the data"""
        if df is None or df.empty or len(df) > 100:  # Skip for large datasets
            return None
        
        viz_config = {}
        
        # Determine visualization type based on data and query
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) > 0 and len(categorical_cols) > 0:
            # Bar chart for categorical vs numeric
            if len(df) <= 20:
                viz_config = {
                    "type": "bar",
                    "x": categorical_cols[0],
                    "y": numeric_cols[0],
                    "title": f"{numeric_cols[0]} by {categorical_cols[0]}"
                }
        
        elif 'status' in df.columns:
            # Pie chart for status distribution
            status_counts = df['status'].value_counts()
            viz_config = {
                "type": "pie",
                "values": status_counts.values.tolist(),
                "labels": status_counts.index.tolist(),
                "title": "Status Distribution"
            }
        
        elif len(numeric_cols) >= 2:
            # Scatter plot for two numeric variables
            viz_config = {
                "type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": f"{numeric_cols[1]} vs {numeric_cols[0]}"
            }
        
        return viz_config if viz_config else None
    
    def _generate_contextual_insights(self, query: str, sql: str, df: pd.DataFrame, 
                                    memory: ConversationMemory, domain_context: List[str]) -> str:
        """Generate insights considering conversation history"""
        
        if df is None or df.empty:
            return """
            **📊 No Data Found**
            
            Your query didn't return any results. This could mean:
            - The data you're looking for doesn't exist in the database
            - Your search criteria might be too specific
            - There might be data quality issues
            
            **💡 Suggestions:**
            - Try broadening your search criteria
            - Check if similar data exists with different filters
            - Verify the database contains the expected information
            """
        
        # Analyze the data structure and content
        insights = []
        
        # Basic data overview with enhanced metrics
        insights.append(f"**📈 Data Overview:** Found {len(df)} record{'s' if len(df) != 1 else ''}")
        
        # Data quality assessment
        total_cells = len(df) * len(df.columns)
        null_cells = df.isnull().sum().sum()
        data_quality = ((total_cells - null_cells) / total_cells) * 100
        insights.append(f"**🔍 Data Quality:** {data_quality:.1f}% complete")
        
        # Column analysis
        columns = df.columns.tolist()
        if 'first_name' in columns and 'last_name' in columns:
            insights.append(f"**👥 People Data:** This appears to be tenant/person information")
        elif 'property_name' in columns or 'address' in columns:
            insights.append(f"**🏠 Property Data:** This shows property-related information")
        elif 'amount' in columns or 'rent_amount' in columns:
            insights.append(f"**💰 Financial Data:** This contains monetary information")
        elif 'status' in columns:
            if 'status' in df.columns:
                status_counts = df['status'].value_counts()
                insights.append(f"**📊 Status Breakdown:** {dict(status_counts)}")
        
        # Advanced analytics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:2]:  # Analyze first 2 numeric columns
                if 'amount' in col.lower() or 'rent' in col.lower():
                    avg_val = df[col].mean()
                    min_val = df[col].min()
                    max_val = df[col].max()
                    std_val = df[col].std()
                    insights.append(f"**💵 {col.title()}:** Average ${avg_val:,.2f} ± ${std_val:,.2f} (Range: ${min_val:,.2f} - ${max_val:,.2f})")
                elif 'count' in col.lower() or col.lower() in ['bedrooms', 'bathrooms']:
                    avg_val = df[col].mean()
                    insights.append(f"**🏠 {col.title()}:** Average {avg_val:.1f}")
        
        # Business intelligence insights
        query_lower = query.lower()
        if 'overdue' in query_lower or 'late' in query_lower:
            if len(df) > 0:
                total_overdue = df[df.select_dtypes(include=[np.number]).columns[0]].sum() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 0
                insights.append(f"**⚠️ Payment Concern:** {len(df)} overdue payment{'s' if len(df) != 1 else ''} totaling ${total_overdue:,.2f}")
                insights.append(f"**📈 Risk Assessment:** {'High' if len(df) > 10 else 'Medium' if len(df) > 5 else 'Low'} priority for collection efforts")
            else:
                insights.append(f"**✅ Payment Status:** Excellent! No overdue payments found.")
        
        elif 'vacant' in query_lower:
            if len(df) > 0:
                insights.append(f"**🏠 Vacancy Alert:** {len(df)} vacant unit{'s' if len(df) != 1 else ''}")
                estimated_loss = len(df) * 1500  # Assume average rent
                insights.append(f"**💰 Revenue Impact:** Estimated monthly loss: ${estimated_loss:,.2f}")
            else:
                insights.append(f"**✅ Occupancy:** Perfect! 100% occupancy rate.")
        
        elif 'maintenance' in query_lower:
            if len(df) > 0:
                if 'priority' in df.columns:
                    priority_counts = df['priority'].value_counts()
                    insights.append(f"**🔧 Maintenance Priorities:** {dict(priority_counts)}")
                    emergency_count = priority_counts.get('emergency', 0)
                    if emergency_count > 0:
                        insights.append(f"**🚨 Urgent Action:** {emergency_count} emergency ticket{'s' if emergency_count != 1 else ''} requiring immediate attention!")
                
        elif 'tenant' in query_lower:
            if len(df) > 0:
                insights.append(f"**👥 Tenant Portfolio:** Managing {len(df)} tenant{'s' if len(df) != 1 else ''}")
                
                # Tenant distribution analysis
                if len(df) < 10:
                    insights.append("**📈 Growth Opportunity:** Small tenant base - consider expansion strategies")
                elif len(df) > 100:
                    insights.append("**🎯 Scale Management:** Large tenant base - leverage automation tools")
                
                # Satisfaction prediction
                satisfaction_score = min(95, 85 + (len(df) * 0.1))  # Mock calculation
                insights.append(f"**😊 Predicted Satisfaction:** {satisfaction_score:.1f}% based on portfolio size")
        
        # Predictive insights
        if len(memory.turns) > 3:
            insights.append("**🔮 AI Prediction:** Based on your query patterns, you might be interested in lease renewal analysis")
        
        # Comparative analysis with previous queries
        if len(memory.turns) > 1:
            prev_turn = memory.turns[-2]
            if prev_turn.results is not None:
                prev_count = len(prev_turn.results) if hasattr(prev_turn.results, '__len__') else 0
                current_count = len(df)
                change = current_count - prev_count
                if change != 0:
                    insights.append(f"**📊 Trend Alert:** {'+' if change > 0 else ''}{change} change from previous query ({prev_turn.user_query[:30]}...)")
        
        # Combine all insights
        final_insights = "\n\n".join(insights)
        
        # Add actionable recommendations
        recommendations = []
        if len(df) > 0:
            if 'amount' in str(df.columns).lower():
                recommendations.append("💰 **Financial Tracking:** Set up automated alerts for amount thresholds")
            if 'email' in df.columns:
                recommendations.append("📧 **Communication:** Implement bulk email campaigns for efficiency")
            if 'phone' in df.columns:
                recommendations.append("📞 **Contact Management:** Verify and update contact information quarterly")
            if 'status' in df.columns:
                recommendations.append("📊 **Status Monitoring:** Create dashboard widgets for real-time status tracking")
        
        if recommendations:
            final_insights += "\n\n**🎯 Smart Recommendations:**\n" + "\n".join([f"- {rec}" for rec in recommendations])
        
        # Add next steps with AI suggestions
        final_insights += "\n\n**🚀 AI-Powered Next Steps:**"
        if 'tenant' in query_lower:
            final_insights += "\n- 🤖 **Auto-Suggestion:** Schedule quarterly tenant satisfaction surveys\n- 📈 **Optimization:** Analyze lease renewal patterns for retention strategies"
        elif 'property' in query_lower:
            final_insights += "\n- 🔍 **Deep Analysis:** Compare property performance metrics\n- 💡 **Innovation:** Consider IoT sensors for predictive maintenance"
        elif 'payment' in query_lower:
            final_insights += "\n- ⚡ **Automation:** Implement smart payment reminders\n- 📊 **Analytics:** Create payment behavior prediction models"
        else:
            final_insights += "\n- 🧠 **AI Enhancement:** This data could be enhanced with machine learning insights\n- 🔗 **Integration:** Consider connecting with external data sources for richer analysis"
        
        return final_insights
    
    def _generate_followup_suggestions(self, df: pd.DataFrame, query: str) -> List[str]:
        """Generate intelligent follow-up suggestions based on results"""
        
        if df is None or df.empty:
            return [
                "Try adjusting your search criteria",
                "Check data availability in different time periods",
                "Explore related data categories",
                "Review database connectivity"
            ]
        
        suggestions = []
        columns = df.columns.tolist()
        
        # Smart suggestions based on result type and columns
        if 'first_name' in columns or 'last_name' in columns:
            suggestions.extend([
                "Show me their detailed contact information",
                "What are their current lease terms?",
                "Any maintenance requests from these tenants?",
                "Check their payment history"
            ])
        
        if 'property_name' in columns or 'address' in columns:
            suggestions.extend([
                "How many units are in these properties?",
                "What's the current occupancy rate?",
                "Show recent maintenance for these properties",
                "Compare property performance metrics"
            ])
        
        if 'amount' in columns or 'rent_amount' in columns:
            suggestions.extend([
                "Show me the complete payment history",
                "Which payments are currently overdue?",
                "What's the average payment amount?",
                "Create a payment trend analysis"
            ])
        
        if 'status' in columns:
            suggestions.extend([
                "Break down by status categories",
                "Show status change history",
                "Create a status dashboard view"
            ])
        
        # Data size based suggestions
        if len(df) > 20:
            suggestions.append("Show me just the top 10 results")
        elif len(df) > 5:
            suggestions.append("Focus on the most important 5 items")
        
        # Query-specific intelligent suggestions
        if 'count' in query.lower():
            suggestions.append("Show me the detailed breakdown")
        elif 'overdue' in query.lower():
            suggestions.extend([
                "How much total is owed?", 
                "When were these payments originally due?",
                "Show tenant contact info for follow-up"
            ])
        elif 'vacant' in query.lower():
            suggestions.extend([
                "How long have they been vacant?", 
                "What's the market rent for these units?",
                "Show similar units that are occupied"
            ])
        elif 'maintenance' in query.lower():
            suggestions.extend([
                "Show by priority level",
                "Which technicians are assigned?",
                "Estimate completion timeline"
            ])
        
        # Advanced AI suggestions
        suggestions.extend([
            "Create a visual dashboard for this data",
            "Export to Excel for detailed analysis",
            "Set up automated alerts for changes"
        ])
        
        return suggestions[:6]  # Limit to 6 suggestions for better UX
    
    def _handle_trend_analysis(self, query: str, db_path: str, memory: ConversationMemory,
                             domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle trend analysis with advanced analytics"""
        
        trend_prompt = f"""
        Analyze trends for: {query}
        Context: {' '.join(domain_context[:2])}
        
        Generate SQL for trend analysis including time-based grouping.
        Focus on patterns over time periods (monthly, quarterly, yearly).
        Include relevant date fields and aggregations.
        
        Return only valid SQLite SQL.
        """
        
        try:
            sql_response = self.model.generate_content(trend_prompt)
            sql = self._clean_sql(sql_response.text)
            
            # Execute query
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            # Generate trend insights
            insights = self._generate_trend_insights(df, query)
            
            # Create trend visualization
            visualization = self._create_trend_visualization(df)
            
            return {
                "type": "trend_analysis",
                "sql": sql,
                "results": df,
                "insights": insights,
                "visualization": visualization,
                "follow_up_suggestions": [
                    "Predict future trends",
                    "Compare with industry benchmarks",
                    "Create automated trend alerts",
                    "Export trend report"
                ],
                "success": True,
                "confidence_score": 0.85
            }
            
        except Exception as e:
            return {
                "type": "trend_analysis",
                "error": str(e),
                "response": f"Trend analysis failed: {str(e)}",
                "success": False,
                "confidence_score": 0.1
            }
    
    def _generate_trend_insights(self, df: pd.DataFrame, query: str) -> str:
        """Generate insights for trend analysis"""
        if df is None or df.empty:
            return "**📈 No trend data available for analysis.**"
        
        insights = []
        insights.append(f"**📊 Trend Analysis:** Analyzing {len(df)} data points")
        
        # Look for time-based columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            main_metric = numeric_cols[0]
            values = df[main_metric].values
            
            if len(values) > 1:
                # Calculate trend direction
                slope = np.polyfit(range(len(values)), values, 1)[0]
                trend_direction = "📈 Increasing" if slope > 0 else "📉 Decreasing" if slope < 0 else "➡️ Stable"
                insights.append(f"**Trend Direction:** {trend_direction}")
                
                # Calculate volatility
                volatility = np.std(values) / np.mean(values) * 100
                volatility_level = "High" if volatility > 20 else "Medium" if volatility > 10 else "Low"
                insights.append(f"**📊 Volatility:** {volatility_level} ({volatility:.1f}%)")
        
        return "\n\n".join(insights)
    
    def _create_trend_visualization(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create trend visualization configuration"""
        if df is None or df.empty:
            return None
        
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            return {
                "type": "line",
                "x": date_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} Trend Over Time"
            }
        
        return None
    
    def _handle_general_query(self, query: str, db_path: str, memory: ConversationMemory,
                            domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle general queries with enhanced AI responses"""
        
        general_prompt = f"""
        Query: {query}
        Domain context: {' '.join(domain_context)}
        Conversation context: {' '.join(conversation_context[:2])}
        User session history: {len(memory.turns)} previous interactions
        
        Provide a helpful, professional response about property management.
        If this could involve data analysis, suggest specific queries.
        Include actionable insights and next steps.
        """
        
        try:
            response = self.model.generate_content(general_prompt)
            
            # Enhance response with smart suggestions
            enhanced_response = response.text + "\n\n"
            enhanced_response += "**🤖 AI Suggestions:**\n"
            enhanced_response += "- Try asking about specific data: 'How many tenants do we have?'\n"
            enhanced_response += "- Explore trends: 'Show payment trends over the last 6 months'\n"
            enhanced_response += "- Get insights: 'What properties need attention?'"
            
            return {
                "type": "general_query",
                "response": enhanced_response,
                "follow_up_suggestions": [
                    "Show me property overview",
                    "Check tenant status",
                    "Review maintenance tickets",
                    "Analyze financial performance"
                ],
                "success": True,
                "confidence_score": 0.8
            }
        except Exception as e:
            return {
                "type": "general_query",
                "response": f"I encountered an error: {str(e)}",
                "success": False,
                "confidence_score": 0.1
            }
    
    def _clean_sql(self, raw_sql: str) -> str:
        """Clean SQL by removing markdown formatting"""
        sql = re.sub(r"```[^\n]*\n", "", raw_sql)
        sql = re.sub(r"\n```", "", sql)
        return sql.strip()
    
    def get_conversation_summary(self, session_id: str) -> str:
        """Get a summary of the conversation with analytics"""
        if session_id not in self.memory_store:
            return "No conversation history"
        
        memory = self.memory_store[session_id]
        if not memory.turns:
            return "No questions asked yet"
        
        summary_parts = []
        for turn in memory.turns[-5:]:  # Last 5 turns
            summary_parts.append(f"Q: {turn.user_query[:50]}...")
        
        analytics_summary = f" | Analytics: {len(memory.turns)} queries, avg time: {self.analytics['avg_response_time']:.2f}s"
        
        return " | ".join(summary_parts) + analytics_summary
    
    def get_analytics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive analytics for dashboard"""
        return {
            "total_queries": self.analytics['total_queries'],
            "success_rate": (self.analytics['successful_queries'] / max(1, self.analytics['total_queries'])) * 100,
            "avg_response_time": self.analytics['avg_response_time'],
            "query_types": self.analytics['query_types'],
            "active_sessions": len(self.memory_store),
            "memory_usage": sum(len(memory.turns) for memory in self.memory_store.values())
        }

# Enhanced Streamlit UI with Advanced Features
def main():
    st.set_page_config(
        page_title="🏠 AI Property Management Assistant", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "AI-Powered Property Management Assistant with Advanced Analytics"
        }
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .suggestion-button {
        margin: 0.2rem;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="main-header"><h1>🏠 AI Property Management Assistant</h1><p>Powered by Agentic AI, RAG Technology, Advanced Analytics & Conversational Memory</p></div>', unsafe_allow_html=True)
    
    # Initialize session state with enhanced features
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    if 'rag_system' not in st.session_state:
        with st.spinner("🧠 Initializing AI Brain..."):
            st.session_state.rag_system = PropertyRAGSystem()
    
    if 'agent' not in st.session_state:
        with st.spinner("🤖 Booting AI Agent..."):
            st.session_state.agent = PropertyManagementAgent(st.session_state.rag_system)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.header("🔧 Control Center")
        
        # Session info
        st.info(f"**Session ID:** {st.session_state.session_id}")
        
        # Database upload
        db_file = st.file_uploader("📁 Upload SQLite Database", type=["db", "sqlite"])
        if db_file:
            db_path = "/tmp/uploaded.db"
            with open(db_path, "wb") as f:
                f.write(db_file.getbuffer())
            st.success("✅ Database uploaded successfully!")
        else:
            db_path = "database.db"
            st.warning("⚠️ Using default database")
        
        # Analytics Dashboard
        if st.session_state.agent:
            analytics = st.session_state.agent.get_analytics_dashboard()
            st.subheader("📊 AI Analytics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", analytics['total_queries'])
                st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
            with col2:
                st.metric("Avg Response", f"{analytics['avg_response_time']:.2f}s")
                st.metric("Active Sessions", analytics['active_sessions'])
            
            # Query type distribution
            if analytics['query_types']:
                st.subheader("🔍 Query Types")
                for qtype, count in analytics['query_types'].items():
                    st.write(f"• {qtype.replace('_', ' ').title()}: {count}")
        
        st.divider()
        
        # Memory controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Memory", use_container_width=True):
                if st.session_state.session_id in st.session_state.agent.memory_store:
                    del st.session_state.agent.memory_store[st.session_state.session_id]
                st.session_state.conversation_history = []
                st.session_state.current_query = ""
                st.success("Memory cleared!")
                time.sleep(1)
                st.rerun()
        
        with col2:
            if st.button("🆕 New Session", use_container_width=True):
                st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
                st.session_state.conversation_history = []
                st.session_state.current_query = ""
                st.success("New session started!")
                time.sleep(1)
                st.rerun()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        quick_actions = [
            "How many tenants do we have?",
            "Show me vacant units",
            "Check overdue payments",
            "List maintenance tickets",
            "Property performance overview"
        ]
        
        for action in quick_actions:
            if st.button(f"🔹 {action}", key=f"quick_{hash(action)}", use_container_width=True):
                st.session_state.current_query = action
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 AI Conversation")
        
        # Display conversation history with enhanced formatting
        if st.session_state.conversation_history:
            with st.expander("📜 Conversation History", expanded=False):
                for i, (user_msg, ai_response, timestamp, execution_time) in enumerate(st.session_state.conversation_history):
                    st.markdown(f"**🕒 {timestamp.strftime('%H:%M:%S')} ({execution_time:.2f}s) - You:**")
                    st.write(user_msg)
                    st.markdown(f"**🤖 AI Assistant:**")
                    st.write(ai_response[:300] + "..." if len(ai_response) > 300 else ai_response)
                    if i < len(st.session_state.conversation_history) - 1:
                        st.divider()
        
        # Enhanced query input with auto-fill functionality
        query_container = st.container()
        with query_container:
            # Use a unique key that changes when current_query changes
            query_key = f"query_input_{hash(st.session_state.current_query)}_{len(st.session_state.conversation_history)}"
            
            query = st.text_area(
                "💭 Ask your question:",
                placeholder="e.g., 'How many tenants do we have?' then follow up with 'Who are they?'",
                height=120,
                value=st.session_state.current_query,
                key=query_key,
                help="Try asking about tenants, properties, payments, or maintenance. I can handle follow-up questions too!"
            )
            
            # Clear the current_query after it's been used
            if st.session_state.current_query and query == st.session_state.current_query:
                st.session_state.current_query = ""
        
        # Enhanced process button
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            process_btn = st.button("🚀 Ask AI Assistant", type="primary", use_container_width=True)
        with col_btn2:
            clear_btn = st.button("🔄 Clear Input", use_container_width=True)
        with col_btn3:
            if st.button("💡 Get Suggestions", use_container_width=True):
                st.session_state.current_query = "What can you help me with?"
                st.rerun()
        
        if clear_btn:
            st.session_state.current_query = ""
            st.rerun()
        
        # Process query with enhanced UI
        if process_btn and query:
            with st.spinner("🤔 AI is thinking..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                result = st.session_state.agent.process_query(
                    query, 
                    db_path, 
                    st.session_state.session_id
                )
                progress_bar.empty()
            
            # Add to conversation history with execution time
            timestamp = datetime.now()
            ai_response = result.get('response', result.get('insights', 'Processed successfully'))
            execution_time = result.get('execution_time', 0)
            st.session_state.conversation_history.append((query, ai_response, timestamp, execution_time))
            
            # Display results with enhanced formatting
            if result["success"]:
                
                # Show execution metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("⚡ Response Time", f"{execution_time:.2f}s")
                with col_metric2:
                    st.metric("🎯 Confidence", f"{result.get('confidence_score', 1.0)*100:.0f}%")
                with col_metric3:
                    st.metric("📝 Query Type", result["type"].replace('_', ' ').title())
                
                # Show context awareness
                if result.get("context_connection"):
                    st.info(f"🔗 {result['context_connection']}")
                
                # Handle different result types
                if result["type"] in ["followup_question", "sql_query"]:
                    st.subheader("📊 Query Results" if result["type"] == "sql_query" else "🔄 Follow-up Response")
                    
                    # Show SQL with syntax highlighting
                    if result.get("sql"):
                        with st.expander("🔍 Generated SQL Query", expanded=False):
                            st.code(result["sql"], language="sql")
                    
                    # Show results with enhanced table
                    if result.get("results") is not None and not result["results"].empty:
                        df = result["results"]
                        st.dataframe(
                            df, 
                            use_container_width=True,
                            height=min(400, (len(df) + 1) * 35)
                        )
                        
                        # Enhanced metrics
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("📋 Rows", len(df))
                        with col_b:
                            st.metric("📊 Columns", len(df.columns))
                        with col_c:
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                avg_val = df[numeric_cols[0]].mean()
                                st.metric(f"📈 Avg {numeric_cols[0]}", f"{avg_val:.2f}")
                        with col_d:
                            # Data completeness
                            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                            st.metric("✅ Complete", f"{completeness:.1f}%")
                        
                        # Visualization
                        if result.get("visualization"):
                            st.subheader("📈 Data Visualization")
                            viz_config = result["visualization"]
                            
                            if viz_config["type"] == "bar":
                                fig = px.bar(df, x=viz_config["x"], y=viz_config["y"], title=viz_config["title"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif viz_config["type"] == "pie":
                                fig = px.pie(values=viz_config["values"], names=viz_config["labels"], title=viz_config["title"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif viz_config["type"] == "scatter":
                                fig = px.scatter(df, x=viz_config["x"], y=viz_config["y"], title=viz_config["title"])
                                st.plotly_chart(fig, use_container_width=True)
                            elif viz_config["type"] == "line":
                                fig = px.line(df, x=viz_config["x"], y=viz_config["y"], title=viz_config["title"])
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download options
                        col_dl1, col_dl2, col_dl3 = st.columns(3)
                        with col_dl1:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📥 Download CSV",
                                csv,
                                f"results_{timestamp.strftime('%Y%m%d_%H%M%S')}.csv",
                                "text/csv",
                                use_container_width=True
                            )
                        with col_dl2:
                            json_data = df.to_json(orient='records', indent=2)
                            st.download_button(
                                "📋 Download JSON",
                                json_data,
                                f"results_{timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                                "application/json",
                                use_container_width=True
                            )
                        with col_dl3:
                            # Create Excel file in memory
                            import io
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                df.to_excel(writer, sheet_name='Results', index=False)
                            excel_data = excel_buffer.getvalue()
                            st.download_button(
                                "📊 Download Excel",
                                excel_data,
                                f"results_{timestamp.strftime('%Y%m%d_%H%M%S')}.xlsx",
                                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    else:
                        st.warning("⚠️ No results found for your query.")
                        st.info("💡 Try adjusting your search criteria or ask a different question.")
                    
                    # Show AI insights
                    if result.get("insights"):
                        st.subheader("🧠 AI Insights & Analytics")
                        st.markdown(result["insights"])
                
                elif result["type"] == "trend_analysis":
                    st.subheader("📈 Trend Analysis")
                    
                    if result.get("visualization"):
                        viz_config = result["visualization"]
                        fig = px.line(result["results"], x=viz_config["x"], y=viz_config["y"], title=viz_config["title"])
                        fig.update_layout(showlegend=True, hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if result.get("insights"):
                        st.markdown(result["insights"])
                
                elif result["type"] == "general_query":
                    st.subheader("💬 AI Response")
                    st.markdown(result["response"])
                
                # Enhanced follow-up suggestions
                if result.get("follow_up_suggestions"):
                    st.subheader("🎯 Smart Follow-up Questions")
                    st.markdown("*Click any suggestion to automatically fill the query box:*")
                    
                    # Create a grid layout for suggestions
                    cols = st.columns(2)
                    for i, suggestion in enumerate(result["follow_up_suggestions"]):
                        with cols[i % 2]:
                            # Create unique button key with timestamp to avoid conflicts
                            button_key = f"suggest_{hash(suggestion)}_{timestamp.strftime('%H%M%S%f')}"
                            if st.button(
                                f"🔹 {suggestion}", 
                                key=button_key, 
                                use_container_width=True,
                                help=f"Click to ask: {suggestion}"
                            ):
                                st.session_state.current_query = suggestion
                                st.rerun()
            
            else:
                st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
                if result.get("sql"):
                    st.code(result["sql"], language="sql")
                
                # Error recovery suggestions
                st.subheader("🔧 Try These Instead:")
                recovery_suggestions = [
                    "Check database connection",
                    "Verify table names exist",
                    "Try a simpler query",
                    "Ask for help with syntax"
                ]
                for suggestion in recovery_suggestions:
                    if st.button(suggestion, key=f"recovery_{hash(suggestion)}"):
                        st.session_state.current_query = suggestion
                        st.rerun()
    
    with col2:
        st.header("📊 Dashboard")
        
        # Real-time status
        memory = st.session_state.agent.get_or_create_memory(st.session_state.session_id)
        
        # Session stats
        st.subheader("📈 Session Statistics")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("💬 Questions Asked", len(memory.turns))
        with col_stat2:
            st.metric("🧠 Memory Items", len(memory.turns) * 3)  # Approximate
        
        # Query type pie chart
        if st.session_state.agent.analytics['query_types']:
            st.subheader("🔍 Query Distribution")
            query_data = st.session_state.agent.analytics['query_types']
            fig_pie = px.pie(
                values=list(query_data.values()),
                names=[name.replace('_', ' ').title() for name in query_data.keys()],
                title="Query Types Distribution"
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Recent activity
        st.subheader("⏰ Recent Activity")
        if memory.turns:
            for turn in memory.turns[-3:]:  # Show last 3
                with st.expander(f"🕒 {turn.timestamp.strftime('%H:%M')} - {turn.query_type.value.title()}", expanded=False):
                    st.write(f"**Q:** {turn.user_query[:100]}...")
                    st.write(f"**Time:** {turn.execution_time:.2f}s")
                    st.write(f"**Confidence:** {turn.confidence_score*100:.0f}%")
        else:
            st.info("💭 No recent activity. Start asking questions!")
        
        # System health
        st.subheader("🔋 System Health")
        analytics = st.session_state.agent.get_analytics_dashboard()
        
        # Success rate indicator
        success_rate = analytics['success_rate']
        if success_rate >= 90:
            st.success(f"✅ Excellent: {success_rate:.1f}% success rate")
        elif success_rate >= 70:
            st.warning(f"⚠️ Good: {success_rate:.1f}% success rate")
        else:
            st.error(f"❌ Needs attention: {success_rate:.1f}% success rate")
        
        # Performance metrics
        avg_time = analytics['avg_response_time']
        if avg_time < 2:
            st.success(f"⚡ Fast: {avg_time:.2f}s avg response")
        elif avg_time < 5:
            st.warning(f"🐌 Moderate: {avg_time:.2f}s avg response")
        else:
            st.error(f"🚨 Slow: {avg_time:.2f}s avg response")
        
        # Memory usage
        memory_usage = analytics['memory_usage']
        memory_health = "🟢 Optimal" if memory_usage < 100 else "🟡 High" if memory_usage < 200 else "🔴 Critical"
        st.info(f"💾 Memory: {memory_health} ({memory_usage} items)")
        
        # AI recommendations
        st.subheader("🤖 AI Recommendations")
        recommendations = []
        
        if len(memory.turns) == 0:
            recommendations.append("🎯 Start with: 'How many tenants do we have?'")
        elif len(memory.turns) < 5:
            recommendations.append("📈 Try asking about trends and analytics")
        else:
            recommendations.append("🔍 Explore advanced queries and filters")
        
        if analytics['success_rate'] < 80:
            recommendations.append("💡 Check database connectivity")
        
        if analytics['avg_response_time'] > 3:
            recommendations.append("⚡ Consider optimizing database queries")
        
        for rec in recommendations:
            st.info(rec)
        
        # Help section
        st.subheader("❓ Quick Help")
        help_items = {
            "📊 Data Queries": "Ask about tenants, properties, payments, maintenance",
            "🔄 Follow-ups": "Ask 'who are they?' after count queries",
            "📈 Trends": "Request 'trends over time' for analytics",
            "💡 Suggestions": "Use the suggestion buttons for quick queries",
            "🔍 Filters": "Add specific criteria to narrow results"
        }
        
        for title, description in help_items.items():
            with st.expander(title):
                st.write(description)
        
        # Export conversation
        if st.button("📤 Export Session", use_container_width=True):
            session_data = {
                "session_id": st.session_state.session_id,
                "timestamp": datetime.now().isoformat(),
                "conversation_history": [
                    {
                        "query": item[0],
                        "response": item[1],
                        "timestamp": item[2].isoformat(),
                        "execution_time": item[3]
                    } for item in st.session_state.conversation_history
                ],
                "analytics": analytics
            }
            
            json_data = json.dumps(session_data, indent=2)
            st.download_button(
                "📋 Download Session JSON",
                json_data,
                f"session_{st.session_state.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()
