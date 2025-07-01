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
    """Enhanced agentic AI system with comprehensive memory"""
    
    def __init__(self, rag_system: PropertyRAGSystem):
        self.rag_system = rag_system
        self.entity_extractor = EntityExtractor()
        self.context_resolver = ContextResolver()
        self.memory_store = {}  # Session ID -> ConversationMemory
        
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
                conversation_summary=""
            )
        return self.memory_store[session_id]
    
    def process_query(self, user_query: str, db_path: str, session_id: str = "default") -> Dict[str, Any]:
        """Main processing pipeline with memory integration"""
        
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
        
        # Process based on type
        if query_type == QueryType.FOLLOWUP_QUESTION:
            result = self._handle_followup_question(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.SQL_QUERY:
            result = self._handle_sql_query(resolved_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.TREND_ANALYSIS:
            result = self._handle_trend_analysis(resolved_query, db_path, memory, domain_context, conversation_context)
        else:
            result = self._handle_general_query(resolved_query, db_path, memory, domain_context, conversation_context)
        
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
            follow_up_suggestions=result.get('follow_up_suggestions', [])
        )
        
        # Update memory
        memory.turns.append(turn)
        if 'results' in result and isinstance(result['results'], pd.DataFrame):
            memory.last_query_results = result['results']
        
        # Add to RAG system
        self.rag_system.add_conversation_turn(turn)
        
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
    
    def _generate_contextual_insights(self, query: str, sql: str, df: pd.DataFrame, 
                                    memory: ConversationMemory, domain_context: List[str]) -> str:
        """Generate insights considering conversation history"""
        
        conversation_summary = ""
        if len(memory.turns) > 1:
            recent_queries = [turn.user_query for turn in memory.turns[-3:]]
            conversation_summary = f"Recent conversation: {' → '.join(recent_queries)}"
        
        insight_prompt = f"""
        Current query: {query}
        SQL: {sql}
        Results: {len(df) if df is not None else 0} rows
        Sample data: {df.head(3).to_dict() if df is not None and not df.empty else 'No results'}
        
        {conversation_summary}
        Domain context: {' '.join(domain_context[:1])}
        
        Provide contextual business insights that:
        1. Address the current query
        2. Connect to previous conversation if relevant
        3. Suggest actionable next steps
        4. Highlight any patterns or concerns
        
        Keep it conversational and practical.
        """
        
        try:
            response = self.model.generate_content(insight_prompt)
            return response.text
        except:
            return "Unable to generate insights at this time."
    
    def _generate_followup_suggestions(self, df: pd.DataFrame, query: str) -> List[str]:
        """Generate intelligent follow-up suggestions based on results"""
        
        if df is None or df.empty:
            return ["Try a different search criteria", "Check if the data exists in the database"]
        
        suggestions = []
        columns = df.columns.tolist()
        
        # Suggestions based on result type and columns
        if 'first_name' in columns or 'last_name' in columns:
            suggestions.extend([
                "Show me their contact information",
                "What are their lease details?",
                "Any maintenance requests from these tenants?"
            ])
        
        if 'property_name' in columns or 'address' in columns:
            suggestions.extend([
                "How many units are in these properties?",
                "What's the occupancy rate?",
                "Show me recent maintenance for these properties"
            ])
        
        if 'amount' in columns or 'rent_amount' in columns:
            suggestions.extend([
                "Show me the payment history",
                "Which payments are overdue?",
                "What's the average amount?"
            ])
        
        if len(df) > 10:
            suggestions.append("Show me just the top 5 results")
        
        # Query-specific suggestions
        if 'count' in query.lower():
            suggestions.append("Show me the detailed list")
        elif 'overdue' in query.lower():
            suggestions.extend(["How much is owed in total?", "When were these due?"])
        elif 'vacant' in query.lower():
            suggestions.extend(["How long have they been vacant?", "What's the asking rent?"])
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _handle_trend_analysis(self, query: str, db_path: str, memory: ConversationMemory,
                             domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle trend analysis with memory context"""
        # Implementation similar to previous version but with memory integration
        return {
            "type": "trend_analysis",
            "response": "Trend analysis with memory integration - implementation in progress",
            "success": True
        }
    
    def _handle_general_query(self, query: str, db_path: str, memory: ConversationMemory,
                            domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle general queries with context"""
        
        general_prompt = f"""
        Query: {query}
        Domain context: {' '.join(domain_context)}
        Conversation context: {' '.join(conversation_context[:2])}
        
        Provide a helpful response about property management.
        If this seems like it should involve data, suggest a specific query.
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

# Enhanced Streamlit UI with Memory
def main():
    st.set_page_config(page_title="🏠 AI Property Management Assistant", layout="wide")
    
    st.title("🏠 AI Property Management Assistant")
    st.markdown("*Powered by Agentic AI, RAG Technology, and Conversational Memory*")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = PropertyRAGSystem()
    
    if 'agent' not in st.session_state:
        st.session_state.agent = PropertyManagementAgent(st.session_state.rag_system)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar
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
        
        # Memory controls
        if st.button("🗑️ Clear Memory"):
            if st.session_state.session_id in st.session_state.agent.memory_store:
                del st.session_state.agent.memory_store[st.session_state.session_id]
            st.session_state.conversation_history = []
            st.rerun()
        
        if st.button("🆕 New Session"):
            st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main interface - single column layout
    st.header("💬 Conversation")
    
    # Display conversation history
    if st.session_state.conversation_history:
        with st.expander("📜 Conversation History", expanded=False):
            for i, (user_msg, ai_response, timestamp) in enumerate(st.session_state.conversation_history):
                st.write(f"**{timestamp.strftime('%H:%M:%S')} - You:** {user_msg}")
                st.write(f"**AI:** {ai_response[:200]}...")
                if i < len(st.session_state.conversation_history) - 1:
                    st.divider()
    
    # Query input
    # Check if there's a suggested query to pre-fill
    default_value = st.session_state.get('suggested_query', '')
    if default_value:
        # Clear the suggested query after using it
        del st.session_state.suggested_query
    
    query = st.text_area(
        "Ask your question:",
        placeholder="e.g., 'How many tenants do we have?' then follow up with 'Who are they?'",
        height=100,
        value=default_value,
        key="current_query"
    )
    
    # Process query
    if st.button("🚀 Ask Question", type="primary", use_container_width=True) and query:
        with st.spinner("🤔 Processing your question..."):
            result = st.session_state.agent.process_query(
                query, 
                db_path, 
                st.session_state.session_id
            )
        
        # Add to conversation history
        timestamp = datetime.now()
        ai_response = result.get('response', result.get('insights', 'Processed successfully'))
        st.session_state.conversation_history.append((query, ai_response, timestamp))
        
        # Display results based on type
        if result["success"]:
            
            # Show context awareness
            if result.get("context_connection"):
                st.info(f"🔗 {result['context_connection']}")
            
            if result["type"] == "followup_question":
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
                        if result["results"].select_dtypes(include=[np.number]).columns.any():
                            numeric_cols = result["results"].select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                avg_val = result["results"][numeric_cols[0]].mean()
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
                    st.subheader("💡 AI Insights")
                    st.markdown(result["insights"])
            
            elif result["type"] == "general_query":
                st.subheader("💬 AI Response")
                st.markdown(result["response"])
            
            # Show follow-up suggestions for ALL successful results
            if result.get("follow_up_suggestions"):
                st.subheader("🎯 Suggested Follow-up Questions")
                for i, suggestion in enumerate(result["follow_up_suggestions"]):
                    # Create a unique key for each suggestion button
                    suggestion_key = f"suggestion_{st.session_state.session_id}_{i}_{len(st.session_state.conversation_history)}"
                    if st.button(f"🔹 {suggestion}", key=suggestion_key):
                        # Set the suggestion in session state and rerun to update the text area
                        st.session_state.suggested_query = suggestion
                        st.rerun()
        
        else:
            st.error(f"❌ Error: {result.get('error', 'Unknown error occurred')}")
            if result.get("sql"):
                st.code(result["sql"], language="sql")

if __name__ == "__main__":
    main()
