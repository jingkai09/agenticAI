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
    CHURN_PREDICTION = "churn_prediction"
    RECOMMENDATION_SYSTEM = "recommendation_system"
    CUSTOMER_SEGMENTATION = "customer_segmentation"
    LIFETIME_VALUE = "lifetime_value"

class PredictionType(Enum):
    OCCUPANCY_FORECAST = "occupancy_forecast"
    RENT_TREND = "rent_trend"
    MAINTENANCE_PREDICTION = "maintenance_prediction"
    CASH_FLOW_PROJECTION = "cash_flow_projection"
    TENANT_CHURN = "tenant_churn"
    MARKET_ANALYSIS = "market_analysis"
    CUSTOMER_CHURN = "customer_churn"
    PURCHASE_INTENTION = "purchase_intention"
    RENEWAL_PROBABILITY = "renewal_probability"
    CROSS_SELL_OPPORTUNITY = "cross_sell_opportunity"
    CUSTOMER_LIFETIME_VALUE = "customer_lifetime_value"
    PRICE_SENSITIVITY = "price_sensitivity"

class CustomerSegment(Enum):
    PREMIUM = "premium"
    STANDARD = "standard"
    BUDGET = "budget"
    AT_RISK = "at_risk"
    HIGH_VALUE = "high_value"
    GROWTH_POTENTIAL = "growth_potential"
    LOYAL = "loyal"
    NEW_CUSTOMER = "new_customer"

@dataclass
class ChurnPrediction:
    """Customer churn prediction results"""
    customer_id: str
    churn_probability: float  # 0-1 scale
    risk_level: str  # "low", "medium", "high", "critical"
    key_risk_factors: List[str]
    retention_recommendations: List[str]
    time_to_churn: Optional[str]  # e.g., "2-3 months"
    confidence_score: float

@dataclass
class RecommendationResult:
    """Recommendation system results"""
    customer_id: str
    recommended_actions: List[str]
    cross_sell_opportunities: List[str]
    upsell_opportunities: List[str]
    optimal_pricing: Optional[float]
    engagement_strategies: List[str]
    purchase_probability: float
    confidence_score: float

@dataclass
class CustomerInsight:
    """Comprehensive customer analytics"""
    customer_id: str
    segment: CustomerSegment
    lifetime_value: float
    predicted_ltv: float
    satisfaction_score: float
    engagement_level: str
    payment_behavior: str
    preferences: Dict[str, Any]
    behavioral_patterns: List[str]

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
    churn_predictions: Optional[List[ChurnPrediction]] = None
    recommendations_data: Optional[List[RecommendationResult]] = None
    customer_insights: Optional[List[CustomerInsight]] = None

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

class CustomerAnalytics:
    """Advanced customer analytics and machine learning engine"""
    
    def __init__(self):
        self.feature_weights = {
            'payment_history': 0.25,
            'engagement_score': 0.20,
            'satisfaction_indicators': 0.20,
            'usage_patterns': 0.15,
            'support_interactions': 0.10,
            'demographic_factors': 0.10
        }
        self.churn_thresholds = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.85
        }
    
    def predict_customer_churn(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame, 
                              tickets_df: pd.DataFrame) -> List[ChurnPrediction]:
        """Predict customer churn probability using multiple data sources"""
        churn_predictions = []
        
        try:
            # Merge customer data with payment and support history
            customer_metrics = self._calculate_customer_metrics(customers_df, payments_df, tickets_df)
            
            for _, customer in customer_metrics.iterrows():
                churn_score = self._calculate_churn_score(customer)
                risk_level = self._determine_risk_level(churn_score)
                risk_factors = self._identify_risk_factors(customer)
                
                prediction = ChurnPrediction(
                    customer_id=str(customer.get('customer_id', customer.get('id', 'unknown'))),
                    churn_probability=churn_score,
                    risk_level=risk_level,
                    key_risk_factors=risk_factors,
                    retention_recommendations=self._generate_retention_strategies(customer, risk_level),
                    time_to_churn=self._estimate_time_to_churn(churn_score),
                    confidence_score=self._calculate_prediction_confidence(customer)
                )
                
                churn_predictions.append(prediction)
        
        except Exception as e:
            # Return a sample prediction with error info
            churn_predictions.append(ChurnPrediction(
                customer_id="error",
                churn_probability=0.5,
                risk_level="unknown",
                key_risk_factors=[f"Error in analysis: {str(e)}"],
                retention_recommendations=["Review data quality and completeness"],
                time_to_churn="unknown",
                confidence_score=0.1
            ))
        
        return sorted(churn_predictions, key=lambda x: x.churn_probability, reverse=True)
    
    def generate_recommendations(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame,
                               leases_df: pd.DataFrame, tickets_df: pd.DataFrame) -> List[RecommendationResult]:
        """Generate personalized recommendations and cross-sell opportunities"""
        recommendations = []
        
        try:
            # Calculate customer profiles
            customer_profiles = self._build_customer_profiles(customers_df, payments_df, leases_df, tickets_df)
            
            for _, customer in customer_profiles.iterrows():
                rec_result = RecommendationResult(
                    customer_id=str(customer.get('customer_id', customer.get('id', 'unknown'))),
                    recommended_actions=self._generate_action_recommendations(customer),
                    cross_sell_opportunities=self._identify_cross_sell_opportunities(customer),
                    upsell_opportunities=self._identify_upsell_opportunities(customer),
                    optimal_pricing=self._calculate_optimal_pricing(customer),
                    engagement_strategies=self._suggest_engagement_strategies(customer),
                    purchase_probability=self._calculate_purchase_probability(customer),
                    confidence_score=self._calculate_recommendation_confidence(customer)
                )
                
                recommendations.append(rec_result)
        
        except Exception as e:
            # Return a sample recommendation with error info
            recommendations.append(RecommendationResult(
                customer_id="error",
                recommended_actions=[f"Error in analysis: {str(e)}"],
                cross_sell_opportunities=["Review data structure"],
                upsell_opportunities=["Ensure data completeness"],
                optimal_pricing=None,
                engagement_strategies=["Fix data pipeline"],
                purchase_probability=0.5,
                confidence_score=0.1
            ))
        
        return sorted(recommendations, key=lambda x: x.purchase_probability, reverse=True)
    
    def segment_customers(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame) -> List[CustomerInsight]:
        """Segment customers and provide insights"""
        customer_insights = []
        
        try:
            # Calculate comprehensive customer metrics
            customer_data = self._calculate_comprehensive_metrics(customers_df, payments_df)
            
            for _, customer in customer_data.iterrows():
                insight = CustomerInsight(
                    customer_id=str(customer.get('customer_id', customer.get('id', 'unknown'))),
                    segment=self._determine_customer_segment(customer),
                    lifetime_value=customer.get('current_ltv', 0),
                    predicted_ltv=customer.get('predicted_ltv', 0),
                    satisfaction_score=customer.get('satisfaction_score', 0.5),
                    engagement_level=self._calculate_engagement_level(customer),
                    payment_behavior=self._analyze_payment_behavior(customer),
                    preferences=self._extract_preferences(customer),
                    behavioral_patterns=self._identify_behavioral_patterns(customer)
                )
                
                customer_insights.append(insight)
        
        except Exception as e:
            # Return a sample insight with error info
            customer_insights.append(CustomerInsight(
                customer_id="error",
                segment=CustomerSegment.STANDARD,
                lifetime_value=0,
                predicted_ltv=0,
                satisfaction_score=0.5,
                engagement_level="unknown",
                payment_behavior=f"Error: {str(e)}",
                preferences={},
                behavioral_patterns=["Data analysis error"]
            ))
        
        return customer_insights
    
    def _calculate_customer_metrics(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame, 
                                  tickets_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key metrics for churn prediction"""
        metrics = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer.get('id')
            
            # Payment metrics
            customer_payments = payments_df[payments_df['tenant_id'] == customer_id] if 'tenant_id' in payments_df.columns else pd.DataFrame()
            
            # Support ticket metrics
            customer_tickets = tickets_df[tickets_df['raised_by'] == customer_id] if 'raised_by' in tickets_df.columns else pd.DataFrame()
            
            # Calculate metrics
            metrics.append({
                'customer_id': customer_id,
                'payment_frequency': len(customer_payments),
                'avg_payment_amount': customer_payments['amount'].mean() if not customer_payments.empty else 0,
                'late_payments': len(customer_payments[customer_payments['paid_on'].isna()]) if 'paid_on' in customer_payments.columns else 0,
                'support_tickets': len(customer_tickets),
                'last_payment_days': self._days_since_last_payment(customer_payments),
                'tenure_days': self._calculate_tenure(customer),
                'complaint_ratio': len(customer_tickets[customer_tickets['priority'] == 'emergency']) / max(len(customer_tickets), 1) if not customer_tickets.empty else 0
            })
        
        return pd.DataFrame(metrics)
    
    def _calculate_churn_score(self, customer: pd.Series) -> float:
        """Calculate churn probability score (0-1)"""
        score = 0.0
        
        # Payment behavior (25%)
        if customer.get('late_payments', 0) > 2:
            score += 0.15
        if customer.get('last_payment_days', 0) > 60:
            score += 0.10
        
        # Support interaction patterns (20%)
        if customer.get('support_tickets', 0) > 5:
            score += 0.10
        if customer.get('complaint_ratio', 0) > 0.3:
            score += 0.10
        
        # Engagement patterns (20%)
        if customer.get('payment_frequency', 0) < 3:
            score += 0.15
        if customer.get('tenure_days', 365) < 90:
            score += 0.05
        
        # Financial indicators (15%)
        avg_payment = customer.get('avg_payment_amount', 0)
        if avg_payment > 0 and avg_payment < 500:
            score += 0.05
        
        # Random factor for realism (20%)
        base_churn_rate = 0.15  # 15% base churn rate
        score += base_churn_rate
        
        return min(score, 1.0)
    
    def _determine_risk_level(self, churn_score: float) -> str:
        """Determine risk level based on churn score"""
        if churn_score >= self.churn_thresholds['critical']:
            return "critical"
        elif churn_score >= self.churn_thresholds['high']:
            return "high"
        elif churn_score >= self.churn_thresholds['medium']:
            return "medium"
        else:
            return "low"
    
    def _identify_risk_factors(self, customer: pd.Series) -> List[str]:
        """Identify specific risk factors for churn"""
        factors = []
        
        if customer.get('late_payments', 0) > 2:
            factors.append("Multiple late payments")
        if customer.get('last_payment_days', 0) > 60:
            factors.append("No recent payments")
        if customer.get('support_tickets', 0) > 5:
            factors.append("High support ticket volume")
        if customer.get('complaint_ratio', 0) > 0.3:
            factors.append("High complaint rate")
        if customer.get('tenure_days', 365) < 90:
            factors.append("New customer (higher churn risk)")
        if customer.get('payment_frequency', 0) < 3:
            factors.append("Low engagement/payment frequency")
        
        if not factors:
            factors.append("General market churn risk")
        
        return factors
    
    def _generate_retention_strategies(self, customer: pd.Series, risk_level: str) -> List[str]:
        """Generate personalized retention strategies"""
        strategies = []
        
        if risk_level == "critical":
            strategies.extend([
                "Immediate personal outreach by account manager",
                "Offer personalized incentives or discounts",
                "Schedule urgent satisfaction review call"
            ])
        elif risk_level == "high":
            strategies.extend([
                "Proactive customer service check-in",
                "Offer flexible payment terms if needed",
                "Provide additional value-added services"
            ])
        elif risk_level == "medium":
            strategies.extend([
                "Send customer satisfaction survey",
                "Offer loyalty program enrollment",
                "Provide service upgrade options"
            ])
        else:
            strategies.extend([
                "Continue excellent service delivery",
                "Occasional check-in communications",
                "Offer referral incentives"
            ])
        
        # Add specific strategies based on risk factors
        if customer.get('late_payments', 0) > 2:
            strategies.append("Set up automated payment reminders")
        if customer.get('support_tickets', 0) > 5:
            strategies.append("Assign dedicated support representative")
        
        return strategies
    
    def _estimate_time_to_churn(self, churn_score: float) -> str:
        """Estimate time until potential churn"""
        if churn_score >= 0.85:
            return "1-2 months"
        elif churn_score >= 0.7:
            return "2-4 months"
        elif churn_score >= 0.5:
            return "4-8 months"
        else:
            return "8+ months"
    
    def _calculate_prediction_confidence(self, customer: pd.Series) -> float:
        """Calculate confidence in the churn prediction"""
        # Base confidence on data completeness and customer history
        data_completeness = sum([
            1 if customer.get('payment_frequency', 0) > 0 else 0,
            1 if customer.get('tenure_days', 0) > 30 else 0,
            1 if customer.get('support_tickets', 0) >= 0 else 0
        ]) / 3
        
        # Higher confidence for customers with more history
        history_factor = min(customer.get('tenure_days', 30) / 365, 1.0)
        
        return (data_completeness * 0.6 + history_factor * 0.4)

    def _build_customer_profiles(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame,
                               leases_df: pd.DataFrame, tickets_df: pd.DataFrame) -> pd.DataFrame:
        """Build comprehensive customer profiles for recommendations"""
        profiles = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer.get('id')
            
            # Get related data
            customer_payments = payments_df[payments_df['tenant_id'] == customer_id] if 'tenant_id' in payments_df.columns else pd.DataFrame()
            customer_leases = leases_df[leases_df['tenant_id'] == customer_id] if 'tenant_id' in leases_df.columns else pd.DataFrame()
            customer_tickets = tickets_df[tickets_df['raised_by'] == customer_id] if 'raised_by' in tickets_df.columns else pd.DataFrame()
            
            # Calculate profile metrics
            profiles.append({
                'customer_id': customer_id,
                'total_payments': customer_payments['amount'].sum() if not customer_payments.empty else 0,
                'avg_payment': customer_payments['amount'].mean() if not customer_payments.empty else 0,
                'payment_consistency': self._calculate_payment_consistency(customer_payments),
                'current_rent': customer_leases['rent_amount'].iloc[-1] if not customer_leases.empty else 0,
                'lease_renewals': len(customer_leases),
                'satisfaction_indicators': self._calculate_satisfaction_score(customer_tickets),
                'service_usage': len(customer_tickets),
                'tenure_months': self._calculate_tenure_months(customer),
                'property_preferences': self._extract_property_preferences(customer_leases)
            })
        
        return pd.DataFrame(profiles)
    
    def _generate_action_recommendations(self, customer: pd.Series) -> List[str]:
        """Generate specific action recommendations for a customer"""
        actions = []
        
        total_payments = customer.get('total_payments', 0)
        satisfaction = customer.get('satisfaction_indicators', 0.5)
        tenure = customer.get('tenure_months', 0)
        
        if total_payments > 10000:
            actions.append("Offer VIP customer benefits and priority service")
        if satisfaction < 0.4:
            actions.append("Schedule satisfaction improvement consultation")
        if tenure > 12:
            actions.append("Recognize loyalty with appreciation program")
        if customer.get('payment_consistency', 0.5) > 0.8:
            actions.append("Offer automatic payment discount")
        if customer.get('service_usage', 0) == 0:
            actions.append("Introduce available services and amenities")
        
        if not actions:
            actions.append("Maintain current service level and monitor satisfaction")
        
        return actions
    
    def _identify_cross_sell_opportunities(self, customer: pd.Series) -> List[str]:
        """Identify cross-selling opportunities"""
        opportunities = []
        
        if customer.get('tenure_months', 0) > 6:
            opportunities.append("Premium maintenance package")
        if customer.get('satisfaction_indicators', 0.5) > 0.7:
            opportunities.append("Refer-a-friend program with incentives")
        if customer.get('total_payments', 0) > 5000:
            opportunities.append("Property management services for owned properties")
        if customer.get('service_usage', 0) < 2:
            opportunities.append("Concierge services and amenity packages")
        
        opportunities.extend([
            "Renter's insurance partnership",
            "Storage unit rental",
            "Parking space upgrade"
        ])
        
        return opportunities[:4]  # Limit to top 4
    
    def _identify_upsell_opportunities(self, customer: pd.Series) -> List[str]:
        """Identify upselling opportunities"""
        opportunities = []
        
        current_rent = customer.get('current_rent', 0)
        payment_consistency = customer.get('payment_consistency', 0.5)
        
        if payment_consistency > 0.8 and current_rent > 0:
            opportunities.append("Premium unit upgrade with better amenities")
        if customer.get('lease_renewals', 0) > 1:
            opportunities.append("Multi-year lease with benefits")
        if customer.get('satisfaction_indicators', 0.5) > 0.8:
            opportunities.append("Luxury service tier upgrade")
        
        opportunities.extend([
            "Extended lease terms with locked rates",
            "Premium appliance packages",
            "Smart home technology upgrades"
        ])
        
        return opportunities[:3]  # Limit to top 3
    
    def _calculate_optimal_pricing(self, customer: pd.Series) -> Optional[float]:
        """Calculate optimal pricing for the customer"""
        current_rent = customer.get('current_rent', 0)
        payment_consistency = customer.get('payment_consistency', 0.5)
        satisfaction = customer.get('satisfaction_indicators', 0.5)
        
        if current_rent == 0:
            return None
        
        # Adjust pricing based on customer profile
        price_multiplier = 1.0
        
        if payment_consistency > 0.9:
            price_multiplier += 0.05  # Can charge slightly more for reliable payers
        if satisfaction > 0.8:
            price_multiplier += 0.03  # Satisfied customers less price sensitive
        if customer.get('tenure_months', 0) > 12:
            price_multiplier -= 0.02  # Loyalty discount
        
        return round(current_rent * price_multiplier, 2)
    
    def _suggest_engagement_strategies(self, customer: pd.Series) -> List[str]:
        """Suggest engagement strategies"""
        strategies = []
        
        if customer.get('service_usage', 0) == 0:
            strategies.append("Send welcome package with service information")
        if customer.get('satisfaction_indicators', 0.5) > 0.8:
            strategies.append("Request positive review or testimonial")
        if customer.get('tenure_months', 0) > 6:
            strategies.append("Send personalized anniversary message")
        
        strategies.extend([
            "Monthly community newsletter",
            "Seasonal maintenance reminders",
            "Feedback surveys and follow-ups"
        ])
        
        return strategies[:4]
    
    def _calculate_purchase_probability(self, customer: pd.Series) -> float:
        """Calculate probability of additional purchases/upgrades"""
        factors = [
            customer.get('payment_consistency', 0.5),
            customer.get('satisfaction_indicators', 0.5),
            min(customer.get('tenure_months', 0) / 12, 1.0),
            min(customer.get('total_payments', 0) / 10000, 1.0)
        ]
        
        return sum(factors) / len(factors)
    
    def _calculate_recommendation_confidence(self, customer: pd.Series) -> float:
        """Calculate confidence in recommendations"""
        data_quality_factors = [
            1 if customer.get('total_payments', 0) > 0 else 0,
            1 if customer.get('tenure_months', 0) > 1 else 0,
            1 if customer.get('satisfaction_indicators', -1) >= 0 else 0
        ]
        
        return sum(data_quality_factors) / len(data_quality_factors)
    
    # Helper methods for calculations
    def _days_since_last_payment(self, payments_df: pd.DataFrame) -> int:
        """Calculate days since last payment"""
        if payments_df.empty or 'paid_on' not in payments_df.columns:
            return 0
        
        last_payment = payments_df['paid_on'].max()
        if pd.isna(last_payment):
            return 365  # No payments made
        
        try:
            last_date = pd.to_datetime(last_payment)
            return (datetime.now() - last_date).days
        except:
            return 0
    
    def _calculate_tenure(self, customer: pd.Series) -> int:
        """Calculate customer tenure in days"""
        if 'created_at' in customer:
            try:
                created_date = pd.to_datetime(customer['created_at'])
                return (datetime.now() - created_date).days
            except:
                return 365
        return 365
    
    def _calculate_tenure_months(self, customer: pd.Series) -> int:
        """Calculate customer tenure in months"""
        return self._calculate_tenure(customer) // 30
    
    def _calculate_payment_consistency(self, payments_df: pd.DataFrame) -> float:
        """Calculate payment consistency score"""
        if payments_df.empty:
            return 0.0
        
        total_payments = len(payments_df)
        on_time_payments = len(payments_df[payments_df['paid_on'].notna()]) if 'paid_on' in payments_df.columns else total_payments
        
        return on_time_payments / total_payments if total_payments > 0 else 0.0
    
    def _calculate_satisfaction_score(self, tickets_df: pd.DataFrame) -> float:
        """Calculate satisfaction score based on support tickets"""
        if tickets_df.empty:
            return 0.7  # Neutral score if no tickets
        
        total_tickets = len(tickets_df)
        emergency_tickets = len(tickets_df[tickets_df['priority'] == 'emergency']) if 'priority' in tickets_df.columns else 0
        
        # Lower satisfaction if many emergency tickets
        satisfaction = max(0.1, 1.0 - (emergency_tickets / total_tickets))
        return satisfaction
    
    def _extract_property_preferences(self, leases_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract property preferences from lease history"""
        if leases_df.empty:
            return {}
        
        preferences = {
            'avg_rent_range': leases_df['rent_amount'].mean() if 'rent_amount' in leases_df.columns else 0,
            'lease_length_preference': 12  # Default
        }
        
        return preferences
    
    def _calculate_comprehensive_metrics(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive metrics for customer segmentation"""
        metrics = []
        
        for _, customer in customers_df.iterrows():
            customer_id = customer.get('id')
            customer_payments = payments_df[payments_df['tenant_id'] == customer_id] if 'tenant_id' in payments_df.columns else pd.DataFrame()
            
            total_value = customer_payments['amount'].sum() if not customer_payments.empty else 0
            
            metrics.append({
                'customer_id': customer_id,
                'current_ltv': total_value,
                'predicted_ltv': total_value * 1.2,  # Simple prediction
                'satisfaction_score': 0.7 + (total_value / 10000) * 0.2,  # Higher value customers assumed more satisfied
                'payment_frequency': len(customer_payments),
                'avg_payment': customer_payments['amount'].mean() if not customer_payments.empty else 0,
                'tenure_months': self._calculate_tenure_months(customer)
            })
        
        return pd.DataFrame(metrics)
    
    def _determine_customer_segment(self, customer: pd.Series) -> CustomerSegment:
        """Determine customer segment"""

# enhanced_property_agent_with_ml_predictions.py - PART 4
# Customer Analytics Segmentation and Predictive Engine

    def _determine_customer_segment(self, customer: pd.Series) -> CustomerSegment:
        """Determine customer segment"""
        ltv = customer.get('current_ltv', 0)
        satisfaction = customer.get('satisfaction_score', 0.5)
        tenure = customer.get('tenure_months', 0)
        
        if ltv > 15000 and satisfaction > 0.8:
            return CustomerSegment.PREMIUM
        elif satisfaction < 0.3 or ltv < 1000:
            return CustomerSegment.AT_RISK
        elif ltv > 8000:
            return CustomerSegment.HIGH_VALUE
        elif tenure < 3:
            return CustomerSegment.NEW_CUSTOMER
        elif satisfaction > 0.8 and tenure > 12:
            return CustomerSegment.LOYAL
        elif ltv > 5000:
            return CustomerSegment.GROWTH_POTENTIAL
        elif ltv < 3000:
            return CustomerSegment.BUDGET
        else:
            return CustomerSegment.STANDARD
    
    def _calculate_engagement_level(self, customer: pd.Series) -> str:
        """Calculate customer engagement level"""
        frequency = customer.get('payment_frequency', 0)
        satisfaction = customer.get('satisfaction_score', 0.5)
        
        if frequency > 10 and satisfaction > 0.7:
            return "high"
        elif frequency > 5 and satisfaction > 0.5:
            return "medium"
        else:
            return "low"
    
    def _analyze_payment_behavior(self, customer: pd.Series) -> str:
        """Analyze payment behavior pattern"""
        avg_payment = customer.get('avg_payment', 0)
        frequency = customer.get('payment_frequency', 0)
        
        if avg_payment > 2000 and frequency > 8:
            return "premium_regular"
        elif frequency > 10:
            return "frequent_payer"
        elif avg_payment > 1500:
            return "high_value"
        elif frequency < 3:
            return "irregular"
        else:
            return "standard"
    
    def _extract_preferences(self, customer: pd.Series) -> Dict[str, Any]:
        """Extract customer preferences"""
        return {
            'preferred_payment_range': customer.get('avg_payment', 0),
            'engagement_preference': self._calculate_engagement_level(customer),
            'value_tier': 'premium' if customer.get('current_ltv', 0) > 10000 else 'standard'
        }
    
    def _identify_behavioral_patterns(self, customer: pd.Series) -> List[str]:
        """Identify behavioral patterns"""
        patterns = []
        
        if customer.get('payment_frequency', 0) > 8:
            patterns.append("Consistent payment behavior")
        if customer.get('satisfaction_score', 0.5) > 0.8:
            patterns.append("High satisfaction levels")
        if customer.get('tenure_months', 0) > 12:
            patterns.append("Long-term customer loyalty")
        if customer.get('current_ltv', 0) > 8000:
            patterns.append("High-value spending pattern")
        
        if not patterns:
            patterns.append("Standard customer behavior")
        
        return patterns

class PredictiveAnalytics:
    """Enhanced predictive analytics engine with ML capabilities"""
    
    def __init__(self):
        self.seasonal_patterns = {}
        self.trend_cache = {}
        self.customer_analytics = CustomerAnalytics()
    
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
    
    def predict_customer_churn_batch(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame, 
                                   tickets_df: pd.DataFrame) -> Dict[str, Any]:
        """Predict customer churn for all customers"""
        churn_predictions = self.customer_analytics.predict_customer_churn(
            customers_df, payments_df, tickets_df
        )
        
        # Aggregate statistics
        high_risk_count = len([p for p in churn_predictions if p.risk_level in ['high', 'critical']])
        avg_churn_prob = sum([p.churn_probability for p in churn_predictions]) / len(churn_predictions) if churn_predictions else 0
        
        return {
            "total_customers_analyzed": len(churn_predictions),
            "high_risk_customers": high_risk_count,
            "average_churn_probability": round(avg_churn_prob, 3),
            "predicted_monthly_churn": max(1, int(len(churn_predictions) * avg_churn_prob / 12)),
            "churn_predictions": churn_predictions,
            "retention_priority": [p for p in churn_predictions if p.risk_level in ['high', 'critical']][:5]
        }
    
    def generate_customer_recommendations_batch(self, customers_df: pd.DataFrame, payments_df: pd.DataFrame,
                                              leases_df: pd.DataFrame, tickets_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate recommendations for all customers"""
        recommendations = self.customer_analytics.generate_recommendations(
            customers_df, payments_df, leases_df, tickets_df
        )
        
        # Aggregate insights
        high_potential = [r for r in recommendations if r.purchase_probability > 0.7]
        total_revenue_opportunity = sum([r.optimal_pricing or 0 for r in recommendations])
        
        return {
            "total_customers_analyzed": len(recommendations),
            "high_potential_customers": len(high_potential),
            "revenue_optimization_opportunity": round(total_revenue_opportunity, 2),
            "recommendations": recommendations,
            "priority_recommendations": high_potential[:5]
        }

class StrategicThinking:
    """Enhanced strategic thinking and scenario planning engine"""
    
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
            "recommendations": [],
            "customer_insights": {}
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
            
            # Enhanced customer analysis
            if 'tenants' in portfolio_data and 'payments' in portfolio_data:
                customer_analytics = CustomerAnalytics()
                customer_insights = customer_analytics.segment_customers(
                    portfolio_data['tenants'], portfolio_data['payments']
                )
                
                # Analyze customer portfolio
                premium_customers = [c for c in customer_insights if c.segment == CustomerSegment.PREMIUM]
                at_risk_customers = [c for c in customer_insights if c.segment == CustomerSegment.AT_RISK]
                
                analysis["customer_insights"] = {
                    "total_customers": len(customer_insights),
                    "premium_customers": len(premium_customers),
                    "at_risk_customers": len(at_risk_customers),
                    "avg_customer_ltv": sum([c.lifetime_value for c in customer_insights]) / len(customer_insights) if customer_insights else 0
                }
                
                if len(premium_customers) > len(customer_insights) * 0.2:
                    analysis["competitive_advantages"].append("Strong premium customer base")
                
                if len(at_risk_customers) > len(customer_insights) * 0.3:
                    analysis["strategic_risks"].append("High proportion of at-risk customers")
                    analysis["recommendations"].append("Implement customer retention program")
        
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    def scenario_planning(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multiple scenarios and strategic responses"""
        scenarios = {
            "optimistic": {
                "description": "Strong market growth, high demand, improved customer retention",
                "occupancy_change": "+5%",
                "rent_change": "+8%",
                "maintenance_change": "+2%",
                "churn_rate_change": "-30%",
                "customer_acquisition": "+25%",
                "strategy": "Expand portfolio, increase rents gradually, invest in premium services"
            },
            "baseline": {
                "description": "Stable market conditions, moderate customer satisfaction",
                "occupancy_change": "0%",
                "rent_change": "+3%",
                "maintenance_change": "+5%",
                "churn_rate_change": "0%",
                "customer_acquisition": "+5%",
                "strategy": "Focus on operational efficiency, implement retention programs"
            },
            "pessimistic": {
                "description": "Economic downturn, increased competition, higher churn",
                "occupancy_change": "-8%",
                "rent_change": "-2%",
                "maintenance_change": "+10%",
                "churn_rate_change": "+50%",
                "customer_acquisition": "-15%",
                "strategy": "Reduce costs, aggressive retention programs, competitive pricing"
            }
        }
        
        return {
            "scenarios": scenarios,
            "recommended_preparations": [
                "Build cash reserves for at least 6 months operating expenses",
                "Implement predictive churn prevention system",
                "Develop customer segmentation and personalization strategy",
                "Create flexible pricing and service tier options",
                "Establish partnerships for cross-selling opportunities",
                "Invest in customer experience technology"
            ],
            "key_indicators_to_monitor": [
                "Customer churn rate and early warning signals",
                "Net Promoter Score and satisfaction metrics",
                "Customer lifetime value trends",
                "Market rental rate movements",
                "Local employment and demographic changes",
                "Competitor pricing and service offerings"
            ]
        }

# enhanced_property_agent_with_ml_predictions.py - PART 5
# RAG System and Context Management

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
            'they': ['tenants', 'properties', 'units', 'payments', 'customers'],
            'them': ['tenants', 'properties', 'units', 'payments', 'customers'],
            'those': ['tenants', 'properties', 'units', 'tickets', 'customers'],
            'these': ['tenants', 'properties', 'units', 'tickets', 'customers'],
            'it': ['property', 'unit', 'payment', 'ticket', 'customer'],
            'that': ['property', 'unit', 'payment', 'ticket', 'customer']
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
    """Enhanced RAG system with conversation memory and ML knowledge"""
    
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
        """Initialize with enhanced property management domain knowledge including ML"""
        domain_docs = [
            """Property Management Best Practices:
            - Regular property inspections should be conducted quarterly
            - Rent collection should be automated with late fees after 5 days
            - Maintenance requests should be categorized by urgency: Emergency (24hrs), Urgent (3 days), Standard (7 days)
            - Tenant screening should include credit check, employment verification, and references
            - Security deposits typically equal 1-2 months rent depending on local laws
            - Monthly rent-to-income ratio should not exceed 30% for qualified tenants""",
            
            """Customer Churn Prediction and Retention:
            - Churn prediction models analyze payment history, satisfaction scores, and engagement patterns
            - Early warning signals include late payments, increased support tickets, and reduced engagement
            - High-risk customers require immediate intervention with personalized retention strategies
            - Customer segmentation enables targeted marketing and service delivery
            - Lifetime value calculations help prioritize retention efforts and resource allocation
            - Predictive analytics can identify at-risk customers 2-3 months before actual churn""",
            
            """Recommendation Systems and Cross-Selling:
            - Purchase intention models analyze customer behavior and preferences
            - Cross-selling opportunities include insurance, storage, parking, and premium services
            - Upselling strategies focus on unit upgrades, extended leases, and service tiers
            - Personalized pricing based on customer value and payment reliability
            - Engagement strategies should be tailored to customer segments and preferences
            - A/B testing helps optimize recommendation algorithms and conversion rates""",
            
            """Customer Analytics and Segmentation:
            - Premium customers: High LTV, excellent payment history, high satisfaction
            - At-risk customers: Late payments, complaints, low engagement
            - Growth potential: Good payment history but low current spend
            - New customers: Require onboarding and early satisfaction monitoring
            - Customer journey mapping identifies key touchpoints and improvement opportunities
            - Behavioral analytics reveal usage patterns and preference trends""",
            
            """Predictive Analytics for Property Management:
            - Occupancy forecasting uses seasonal patterns, local economic indicators, and historical data
            - Maintenance prediction models analyze equipment age, usage patterns, and failure history
            - Rent optimization considers market comparables, tenant quality, and demand patterns
            - Cash flow forecasting incorporates collection rates, vacancy predictions, and expense trends
            - Market analysis includes demographic trends, employment data, and development patterns
            - Machine learning models improve accuracy with more data and feedback loops""",
            
            """Strategic Planning Frameworks:
            - Portfolio diversification reduces concentration risk across property types and locations
            - Capital allocation should prioritize high-ROI improvements and strategic acquisitions
            - Risk management includes insurance coverage, reserve funds, and scenario planning
            - Market positioning analysis compares amenities, pricing, and service levels
            - Growth strategies may include acquisition, development, or value-add improvements
            - Technology adoption can improve operational efficiency and customer satisfaction""",
            
            """Key Performance Indicators and Benchmarks:
            - Occupancy rate targets: 95-98% for stabilized properties
            - Rent collection rate: 98%+ considered excellent, 95%+ acceptable
            - Customer churn rate: <5% monthly for residential, varies by market
            - Net Promoter Score: 50+ considered good, 70+ excellent
            - Customer lifetime value: Should be 3-5x annual revenue per customer
            - Maintenance cost ratio: 15-25% of gross rental income
            - Net Operating Income (NOI) margins: 60-80% for well-managed properties""",
            
            """Machine Learning Applications in Property Management:
            - Predictive models for maintenance scheduling and cost optimization
            - Dynamic pricing algorithms based on market conditions and customer segments
            - Automated tenant screening using credit, employment, and behavioral data
            - Sentiment analysis of customer feedback and reviews
            - Computer vision for property condition assessment and maintenance needs
            - Natural language processing for automated customer service and support
            - Recommendation engines for personalized service offerings and upgrades"""
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
    """Enhanced agentic AI system with ML predictions and recommendation engine"""
    
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
        You are an expert Property Management AI Agent with advanced machine learning capabilities including customer churn prediction, recommendation systems, and strategic analytics.
        
        CORE CAPABILITIES:
        - Generate and execute SQL queries for property management databases
        - Maintain conversation context and memory across multiple turns
        - Perform predictive analytics and forecasting
        - Conduct strategic analysis and scenario planning
        - Predict customer churn and recommend retention strategies
        - Generate personalized recommendations and cross-selling opportunities
        - Customer segmentation and lifetime value analysis
        - Provide data-driven insights and recommendations
        
        MACHINE LEARNING CAPABILITIES:
        - Customer churn prediction using payment history, satisfaction scores, and engagement patterns
        - Purchase intention modeling for cross-sell and upsell opportunities
        - Customer segmentation based on behavior, value, and risk profiles
        - Lifetime value calculation and prediction
        - Personalized pricing optimization
        - Recommendation engine for services, upgrades, and engagement strategies
        
        CUSTOMER ANALYTICS:
        - Segment customers into Premium, Standard, At-Risk, High-Value, Growth Potential, Loyal, New Customer
        - Predict churn probability with confidence scores and time-to-churn estimates
        - Identify key risk factors and generate personalized retention strategies
        - Recommend optimal pricing, cross-sell opportunities, and engagement approaches
        - Calculate customer lifetime value and predict future value potential
        
        PREDICTIVE ANALYTICS:
        - Occupancy forecasting based on historical patterns and market indicators
        - Maintenance prediction using equipment and category analysis
        - Cash flow projections with risk assessment and collection rate prediction
        - Market trend analysis and competitive positioning
        - Revenue optimization through dynamic pricing and service recommendations
        
        STRATEGIC THINKING:
        - Portfolio optimization and diversification analysis
        - Growth opportunity identification with customer-centric approach
        - Risk assessment and mitigation planning including customer churn risks
        - Scenario planning for different market conditions and customer behavior changes
        - Investment prioritization and resource allocation based on customer value
        
        DATABASE SCHEMA:
        - tenants(id, timestamp, first_name, last_name, email, phone, date_of_birth, created_at)
        - properties(id, timestamp, name, address_line1, address_line2, city, state, postal_code, country, created_at)
        - units(id, timestamp, property_id, unit_number, floor, bedrooms, bathrooms, square_feet, status, created_at)
        - leases(id, timestamp, tenant_id, unit_id, start_date, end_date, rent_amount, security_deposit, status, created_at)
        - payments(id, timestamp, lease_id, payment_type, billing_period, due_date, amount, method, paid_on, reference_number, created_at)
        - service_tickets(id, timestamp, lease_id, raised_by, assigned_to, category, subcategory, description, status, priority, created_at, updated_at)
        
        RESPONSE GUIDELINES:
        - Always explain your reasoning process for complex analyses
        - Provide confidence levels for predictions (high/medium/low)
        - Include assumptions and limitations in your analysis
        - Suggest actionable recommendations based on insights
        - Consider multiple scenarios when appropriate
        - Use data visualization suggestions when helpful
        - For customer analytics, always consider both individual and portfolio-level insights
        - Prioritize customer retention and lifetime value optimization
        
        When performing predictive analysis, always:
        1. Explain your methodology including ML approach
        2. State your assumptions clearly
        3. Provide confidence intervals or levels
        4. Suggest data improvements for better accuracy
        5. Include risk factors and limitations
        6. Consider customer segmentation in recommendations
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
        """Main processing pipeline with enhanced ML capabilities"""
        
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
        
        # Enhanced query classification with ML capabilities
        query_type = self._classify_intent(resolved_query, memory)
        
        # Process based on type with enhanced ML capabilities
        if query_type == QueryType.CHURN_PREDICTION:
            result = self._handle_churn_prediction(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.RECOMMENDATION_SYSTEM:
            result = self._handle_recommendation_system(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.CUSTOMER_SEGMENTATION:
            result = self._handle_customer_segmentation(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        elif query_type == QueryType.LIFETIME_VALUE:
            result = self._handle_lifetime_value_analysis(resolved_query, user_query, db_path, memory, domain_context, conversation_context)
        else:
            # Handle other query types with simplified implementations
            result = self._handle_general_query(resolved_query, db_path, memory, domain_context, conversation_context)
        
        # Create conversation turn with enhanced insights
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
        """Enhanced intent classification including ML capabilities"""
        
        # Keywords for churn prediction
        churn_keywords = [
            'churn', 'leaving', 'cancel', 'retention', 'at risk', 'likely to leave',
            'probability of leaving', 'who might leave', 'customer risk'
        ]
        
        # Keywords for recommendation system
        recommendation_keywords = [
            'recommend', 'suggest', 'cross sell', 'upsell', 'purchase intention',
            'what should we offer', 'opportunities', 'personalized', 'targeting'
        ]
        
        # Keywords for customer segmentation
        segmentation_keywords = [
            'segment', 'customer types', 'categorize customers', 'customer groups',
            'premium customers', 'high value', 'customer analysis'
        ]
        
        # Keywords for lifetime value
        ltv_keywords = [
            'lifetime value', 'ltv', 'customer value', 'long term value',
            'customer worth', 'revenue per customer'
        ]
        
        query_lower = query.lower()
        
        # Check for ML-specific query types first
        if any(keyword in query_lower for keyword in churn_keywords):
            return QueryType.CHURN_PREDICTION
        
        if any(keyword in query_lower for keyword in recommendation_keywords):
            return QueryType.RECOMMENDATION_SYSTEM
        
        if any(keyword in query_lower for keyword in segmentation_keywords):
            return QueryType.CUSTOMER_SEGMENTATION
        
        if any(keyword in query_lower for keyword in ltv_keywords):
            return QueryType.LIFETIME_VALUE
        
        # Default to general query
        return QueryType.GENERAL_QUERY
    
    def _learn_patterns(self, turn: ConversationTurn, memory: ConversationMemory):
        """Enhanced pattern learning including ML insights"""
        
        # Learn user preferences and common query patterns
        if turn.query_type in [QueryType.CHURN_PREDICTION, QueryType.RECOMMENDATION_SYSTEM, 
                              QueryType.CUSTOMER_SEGMENTATION, QueryType.LIFETIME_VALUE]:
            
            # Track what types of ML queries users ask for
            ml_patterns = memory.learned_patterns.get('ml_query_types', [])
            ml_patterns.append(turn.query_type.value)
            memory.learned_patterns['ml_query_types'] = ml_patterns[-10:]  # Keep last 10
            
            # Track entities mentioned in ML queries
            entity_patterns = memory.learned_patterns.get('entities_in_ml', {})
            for entity in turn.entities_mentioned:
                entity_patterns[entity] = entity_patterns.get(entity, 0) + 1
            memory.learned_patterns['entities_in_ml'] = entity_patterns

# enhanced_property_agent_with_ml_predictions.py - PART 7
# ML Query Handlers (continuation of PropertyManagementAgent class)

    def _handle_churn_prediction(self, resolved_query: str, original_query: str, db_path: str, 
                               memory: ConversationMemory, domain_context: List[str], 
                               conversation_context: List[str]) -> Dict[str, Any]:
        """Handle customer churn prediction queries"""
        
        try:
            # Get customer data
            conn = sqlite3.connect(db_path)
            customers_df = pd.read_sql_query("SELECT * FROM tenants", conn)
            payments_df = pd.read_sql_query("SELECT * FROM payments", conn)
            tickets_df = pd.read_sql_query("SELECT * FROM service_tickets", conn)
            conn.close()
            
            # Perform churn prediction
            churn_analysis = self.predictive_engine.predict_customer_churn_batch(
                customers_df, payments_df, tickets_df
            )
            
            # Generate response
            response = f"""## 🚨 Customer Churn Prediction Analysis

**Analysis Question:** {original_query}

### 📊 Churn Risk Overview

**Customers Analyzed:** {churn_analysis['total_customers_analyzed']}
**High-Risk Customers:** {churn_analysis['high_risk_customers']} 
**Average Churn Probability:** {churn_analysis['average_churn_probability']:.1%}
**Predicted Monthly Churn:** {churn_analysis['predicted_monthly_churn']} customers

### 🎯 Priority Customers for Immediate Attention
"""
            
            if churn_analysis.get('retention_priority'):
                for i, customer in enumerate(churn_analysis['retention_priority'][:3], 1):
                    response += f"""
**{i}. Customer {customer.customer_id}**
- 🔴 Churn Risk: {customer.churn_probability:.1%} ({customer.risk_level.upper()})
- ⏰ Estimated Time to Churn: {customer.time_to_churn}
- ⚠️ Key Risk Factors: {', '.join(customer.key_risk_factors[:2])}
- 💡 Top Retention Strategy: {customer.retention_recommendations[0] if customer.retention_recommendations else 'Immediate personal outreach'}
"""
            
            response += """
### 🛡️ Recommended Actions

**Immediate (Next 7 Days):**
- Contact all CRITICAL risk customers with personalized outreach
- Review specific risk factors for HIGH risk customers
- Implement automated early warning system

**Medium-term (Next 30 Days):**
- Launch satisfaction improvement program
- Develop loyalty incentives for medium-risk customers
- Establish regular check-in protocols
"""
            
            # Create results DataFrame for display
            if churn_analysis.get('churn_predictions'):
                churn_df = pd.DataFrame([
                    {
                        'Customer ID': p.customer_id,
                        'Churn Probability': f"{p.churn_probability:.2%}",
                        'Risk Level': p.risk_level.title(),
                        'Time to Churn': p.time_to_churn,
                        'Key Risk Factors': '; '.join(p.key_risk_factors[:2]),
                        'Confidence': f"{p.confidence_score:.2%}"
                    }
                    for p in churn_analysis['churn_predictions'][:10]  # Top 10
                ])
            else:
                churn_df = pd.DataFrame()
            
            suggestions = [
                "Show me retention strategies for high-risk customers",
                "What are the main factors driving churn?",
                "How can we improve customer satisfaction scores?",
                "Generate personalized retention campaigns"
            ]
            
            return {
                "type": "churn_prediction",
                "results": churn_df,
                "response": response,
                "churn_analysis": churn_analysis,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "churn_prediction",
                "error": str(e),
                "response": f"I encountered an error during churn prediction analysis: {str(e)}",
                "success": False
            }
    
    def _handle_recommendation_system(self, resolved_query: str, original_query: str, db_path: str, 
                                    memory: ConversationMemory, domain_context: List[str], 
                                    conversation_context: List[str]) -> Dict[str, Any]:
        """Handle recommendation system and cross-selling queries"""
        
        try:
            # Get comprehensive customer data
            conn = sqlite3.connect(db_path)
            customers_df = pd.read_sql_query("SELECT * FROM tenants", conn)
            payments_df = pd.read_sql_query("SELECT * FROM payments", conn)
            leases_df = pd.read_sql_query("SELECT * FROM leases", conn)
            tickets_df = pd.read_sql_query("SELECT * FROM service_tickets", conn)
            conn.close()
            
            # Generate recommendations
            recommendation_analysis = self.predictive_engine.generate_customer_recommendations_batch(
                customers_df, payments_df, leases_df, tickets_df
            )
            
            # Generate response
            response = f"""## 🎯 Personalized Recommendation Analysis

**Question:** {original_query}

### 📊 Opportunity Overview

**Customers Analyzed:** {recommendation_analysis['total_customers_analyzed']}
**High-Potential Customers:** {recommendation_analysis['high_potential_customers']}
**Revenue Optimization Opportunity:** ${recommendation_analysis['revenue_optimization_opportunity']:,.2f}

### 🚀 Top Revenue Opportunities
"""
            
            if recommendation_analysis.get('priority_recommendations'):
                for i, rec in enumerate(recommendation_analysis['priority_recommendations'][:3], 1):
                    response += f"""
**{i}. Customer {rec.customer_id}**
- 🎯 Purchase Probability: {rec.purchase_probability:.1%}
- 💰 Optimal Pricing: ${rec.optimal_pricing:.2f} if rec.optimal_pricing else "N/A"}
- 🔄 Top Cross-sell: {rec.cross_sell_opportunities[0] if rec.cross_sell_opportunities else 'N/A'}
- ⬆️ Upsell Opportunity: {rec.upsell_opportunities[0] if rec.upsell_opportunities else 'N/A'}
"""
            
            response += """
### 💡 Strategic Recommendations

**Cross-Selling Priorities:**
1. **Premium Maintenance Packages** - High satisfaction correlation
2. **Renter's Insurance Partnerships** - Low effort, steady revenue
3. **Storage and Parking Upgrades** - Natural upsell opportunities
4. **Concierge Services** - Premium tier differentiation

**Implementation Roadmap:**
- **Phase 1:** Contact top 5 high-probability customers
- **Phase 2:** Deploy segmented marketing campaigns  
- **Phase 3:** Optimize based on results and feedback
"""
            
            # Create results DataFrame
            if recommendation_analysis.get('recommendations'):
                rec_df = pd.DataFrame([
                    {
                        'Customer ID': r.customer_id,
                        'Purchase Probability': f"{r.purchase_probability:.2%}",
                        'Top Recommendation': r.recommended_actions[0] if r.recommended_actions else 'N/A',
                        'Cross-sell Opportunity': r.cross_sell_opportunities[0] if r.cross_sell_opportunities else 'N/A',
                        'Optimal Pricing': f"${r.optimal_pricing:.2f}" if r.optimal_pricing else 'N/A',
                        'Confidence': f"{r.confidence_score:.2%}"
                    }
                    for r in recommendation_analysis['recommendations'][:10]  # Top 10
                ])
            else:
                rec_df = pd.DataFrame()
            
            suggestions = [
                "Show me the highest-value opportunities",
                "What are the best cross-selling products?",
                "How should we price these recommendations?",
                "Generate marketing campaigns for these opportunities"
            ]
            
            return {
                "type": "recommendation_system",
                "results": rec_df,
                "response": response,
                "recommendation_analysis": recommendation_analysis,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "recommendation_system",
                "error": str(e),
                "response": f"I encountered an error during recommendation analysis: {str(e)}",
                "success": False
            }
    
    def _handle_customer_segmentation(self, resolved_query: str, original_query: str, db_path: str, 
                                    memory: ConversationMemory, domain_context: List[str], 
                                    conversation_context: List[str]) -> Dict[str, Any]:
        """Handle customer segmentation analysis"""
        
        try:
            # Get customer data
            conn = sqlite3.connect(db_path)
            customers_df = pd.read_sql_query("SELECT * FROM tenants", conn)
            payments_df = pd.read_sql_query("SELECT * FROM payments", conn)
            conn.close()
            
            # Perform customer segmentation
            customer_insights = self.predictive_engine.customer_analytics.segment_customers(
                customers_df, payments_df
            )
            
            # Aggregate segment statistics
            segment_stats = {}
            for insight in customer_insights:
                segment = insight.segment.value
                if segment not in segment_stats:
                    segment_stats[segment] = {
                        'count': 0,
                        'total_ltv': 0,
                        'avg_satisfaction': 0
                    }
                segment_stats[segment]['count'] += 1
                segment_stats[segment]['total_ltv'] += insight.lifetime_value
                segment_stats[segment]['avg_satisfaction'] += insight.satisfaction_score
            
            # Calculate averages
            for segment in segment_stats:
                count = segment_stats[segment]['count']
                if count > 0:
                    segment_stats[segment]['avg_ltv'] = segment_stats[segment]['total_ltv'] / count
                    segment_stats[segment]['avg_satisfaction'] = segment_stats[segment]['avg_satisfaction'] / count
            
            # Generate response
            total_customers = sum([stats['count'] for stats in segment_stats.values()])
            
            response = f"""## 👥 Customer Segmentation Analysis

**Segmentation Question:** {original_query}

### 📊 Customer Portfolio Overview

**Total Customers Analyzed:** {total_customers}
**Segments Identified:** {len(segment_stats)}

### 🎯 Segment Breakdown & Strategies
"""
            
            # Sort segments by total LTV for priority display
            sorted_segments = sorted(segment_stats.items(), key=lambda x: x[1]['total_ltv'], reverse=True)
            
            segment_icons = {
                'premium': '💎',
                'high_value': '🌟',
                'loyal': '❤️',
                'growth_potential': '🚀',
                'standard': '👤',
                'new_customer': '🆕',
                'budget': '💰',
                'at_risk': '⚠️'
            }
            
            for segment, stats in sorted_segments:
                icon = segment_icons.get(segment, '👤')
                percentage = (stats['count'] / total_customers) * 100 if total_customers > 0 else 0
                
                response += f"""
**{icon} {segment.replace('_', ' ').title()} Segment**
- **Size:** {stats['count']} customers ({percentage:.1f}% of portfolio)
- **Avg Lifetime Value:** ${stats['avg_ltv']:,.2f}
- **Avg Satisfaction:** {stats['avg_satisfaction']:.1%}
- **Total Portfolio Value:** ${stats['total_ltv']:,.2f}
"""
            
            response += """
### 💼 Strategic Recommendations by Segment

**Resource Allocation Priorities:**
1. **Premium/High-Value Segments** - Invest in retention and satisfaction
2. **At-Risk Segment** - Immediate intervention and recovery programs
3. **Growth Potential** - Targeted upselling and engagement campaigns
4. **New Customers** - Excellent onboarding and early satisfaction
"""
            
            # Create results DataFrame
            segment_df = pd.DataFrame([
                {
                    'Segment': segment.replace('_', ' ').title(),
                    'Customer Count': stats['count'],
                    'Avg Lifetime Value': f"${stats['avg_ltv']:.2f}",
                    'Avg Satisfaction': f"{stats['avg_satisfaction']:.2%}",
                    'Total Portfolio Value': f"${stats['total_ltv']:.2f}"
                }
                for segment, stats in segment_stats.items()
            ])
            
            suggestions = [
                "Show me detailed strategies for each segment",
                "Which segment has the highest growth potential?",
                "How should we prioritize our marketing budget?",
                "What are the key characteristics of premium customers?"
            ]
            
            return {
                "type": "customer_segmentation",
                "results": segment_df,
                "response": response,
                "segment_stats": segment_stats,
                "customer_insights": customer_insights,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "customer_segmentation",
                "error": str(e),
                "response": f"I encountered an error during customer segmentation: {str(e)}",
                "success": False
            }
    
    def _handle_lifetime_value_analysis(self, resolved_query: str, original_query: str, db_path: str, 
                                      memory: ConversationMemory, domain_context: List[str], 
                                      conversation_context: List[str]) -> Dict[str, Any]:
        """Handle customer lifetime value analysis"""
        
        try:
            # Get customer data
            conn = sqlite3.connect(db_path)
            customers_df = pd.read_sql_query("SELECT * FROM tenants", conn)
            payments_df = pd.read_sql_query("SELECT * FROM payments", conn)
            conn.close()
            
            # Calculate LTV for all customers
            customer_insights = self.predictive_engine.customer_analytics.segment_customers(
                customers_df, payments_df
            )
            
            # Aggregate LTV statistics
            total_ltv = sum([c.lifetime_value for c in customer_insights])
            avg_ltv = total_ltv / len(customer_insights) if customer_insights else 0
            predicted_total_ltv = sum([c.predicted_ltv for c in customer_insights])
            
            # Generate response
            response = f"""## 💰 Customer Lifetime Value Analysis

**LTV Analysis Question:** {original_query}

### 📊 Portfolio Value Overview

**Total Portfolio LTV:** ${total_ltv:,.2f}
**Average Customer LTV:** ${avg_ltv:,.2f}
**Predicted Future LTV:** ${predicted_total_ltv:,.2f}
**Growth Potential:** {((predicted_total_ltv - total_ltv) / total_ltv * 100):+.1f}% if total_ltv > 0 else "N/A"

### 🚀 Value Optimization Strategies

**For High-Value Customers:**
- Implement VIP retention programs with premium services
- Offer exclusive amenities and personalized experiences
- Priority customer service and dedicated account management

**For Growth Opportunity Customers:**
- Analyze barriers to higher engagement and spending
- Implement value demonstration programs
- Focus on satisfaction improvement and retention

### 📈 Revenue Optimization Opportunities

**Immediate Actions (0-30 days):**
1. **High-Value Customer Protection** - Ensure 100% satisfaction for top 20%
2. **Pricing Strategy Review** - Optimize pricing for value segments
3. **Churn Prevention** - Implement early warning systems

**Growth Initiatives (30-90 days):**
1. **Upselling Programs** - Target medium-value customers with 70%+ satisfaction
2. **Cross-selling Campaigns** - Introduce complementary services
3. **Value Migration** - Move customers from low to medium value segments
"""
            
            # Create results DataFrame
            ltv_df = pd.DataFrame([
                {
                    'Customer ID': c.customer_id,
                    'Current LTV': f"${c.lifetime_value:.2f}",
                    'Predicted LTV': f"${c.predicted_ltv:.2f}",
                    'Segment': c.segment.value.replace('_', ' ').title(),
                    'Satisfaction Score': f"{c.satisfaction_score:.2%}",
                    'Value Category': 'High' if c.lifetime_value > avg_ltv * 2 else 'Medium' if c.lifetime_value >= avg_ltv else 'Low'
                }
                for c in sorted(customer_insights, key=lambda x: x.lifetime_value, reverse=True)[:15]  # Top 15
            ])
            
            suggestions = [
                "Which customers have the highest growth potential?",
                "How can we increase average customer lifetime value?",
                "Show me retention strategies for high-value customers",
                "What factors correlate with high lifetime value?"
            ]
            
            return {
                "type": "lifetime_value",
                "results": ltv_df,
                "response": response,
                "customer_insights": customer_insights,
                "follow_up_suggestions": suggestions,
                "success": True
            }
            
        except Exception as e:
            return {
                "type": "lifetime_value",
                "error": str(e),
                "response": f"I encountered an error during lifetime value analysis: {str(e)}",
                "success": False
            }
    
    def _handle_general_query(self, query: str, db_path: str, memory: ConversationMemory,
                            domain_context: List[str], conversation_context: List[str]) -> Dict[str, Any]:
        """Handle general queries with context"""
        
        general_prompt = f"""
        Query: {query}
        Domain context: {' '.join(domain_context)}
        Conversation context: {' '.join(conversation_context[:2])}
        
        Provide a helpful response about property management with ML capabilities.
        If this involves data analysis, suggest a specific approach.
        If it involves predictions or strategic thinking, explain how to approach

        general_prompt = f"""
        Query: {query}
        Domain context: {' '.join(domain_context)}
        Conversation context: {' '.join(conversation_context[:2])}
        
        Provide a helpful response about property management with ML capabilities.
        If this involves data analysis, suggest a specific approach.
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

def enhanced_sidebar_with_ml_settings():
    """Enhanced sidebar with ML and recommendation settings"""
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
        
        # ML Analytics Settings
        st.header("🤖 ML Analytics Settings")
        
        analytics_mode = st.selectbox(
            "Default Analysis Mode",
            ["Basic", "Predictive", "Strategic", "ML-Enhanced", "Comprehensive"],
            index=3,
            help="Choose the default level of analysis for your queries"
        )
        
        confidence_threshold = st.slider(
            "ML Prediction Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Minimum confidence level for displaying ML predictions"
        )
        
        churn_risk_threshold = st.slider(
            "Churn Risk Alert Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Churn probability threshold for high-risk alerts"
        )
        
        show_thinking = st.checkbox(
            "Show AI Reasoning Process",
            value=True,
            help="Display the AI's thinking process and methodology"
        )
        
        st.session_state.analytics_settings = {
            "mode": analytics_mode,
            "confidence_threshold": confidence_threshold,
            "churn_risk_threshold": churn_risk_threshold,
            "show_thinking": show_thinking
        }
        
        st.divider()
        
        # Customer Analytics Dashboard
        st.header("👥 Customer Analytics")
        
        if st.button("🚨 Churn Risk Dashboard", use_container_width=True):
            st.session_state.suggested_query = "Predict which customers are at risk of churning"
            st.rerun()
        
        if st.button("🎯 Recommendation Engine", use_container_width=True):
            st.session_state.suggested_query = "Generate personalized recommendations for all customers"
            st.rerun()
        
        if st.button("📊 Customer Segmentation", use_container_width=True):
            st.session_state.suggested_query = "Analyze customer segments and provide strategic insights"
            st.rerun()
        
        if st.button("💰 Lifetime Value Analysis", use_container_width=True):
            st.session_state.suggested_query = "Calculate customer lifetime value and growth opportunities"
            st.rerun()
        
        st.divider()
        
        # Session Management (condensed)
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
                save_sessions_to_disk(st.session_state.agent)
                st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
                st.session_state.conversation_history = []
                st.success("New session created!")
                st.rerun()
        
        with col2:
            if st.button("💾 Save Sessions", use_container_width=True):
                save_sessions_to_disk(st.session_state.agent)
                st.success("Sessions saved!")
        
        return db_path

def main():
    st.set_page_config(page_title="🏠 AI Property Management Assistant with ML", layout="wide")
    
    st.title("🏠 AI Property Management Assistant")
    st.markdown("*Powered by Machine Learning: Churn Prediction, Recommendation Engine & Advanced Analytics*")
    
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
            "mode": "ML-Enhanced",
            "confidence_threshold": 0.6,
            "churn_risk_threshold": 0.7,
            "show_thinking": True
        }
    
    # Enhanced sidebar with ML settings
    db_path = enhanced_sidebar_with_ml_settings()
    
    # Auto-save sessions periodically
    if len(st.session_state.agent.memory_store) > 0:
        save_sessions_to_disk(st.session_state.agent)
    
    # Main interface with ML capabilities
    st.header("💬 Conversation")
    
    # ML Analytics Overview
    if st.session_state.analytics_settings["mode"] in ["ML-Enhanced", "Comprehensive"]:
        with st.expander("🤖 ML Analytics Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🚨 Churn Risk Threshold", f"{st.session_state.analytics_settings['churn_risk_threshold']:.0%}")
            with col2:
                st.metric("🔍 Confidence Level", f"{st.session_state.analytics_settings['confidence_threshold']:.0%}")
            with col3:
                st.metric("🧠 AI Mode", st.session_state.analytics_settings["mode"])
            with col4:
                st.metric("📊 ML Features", "Active")
    
    # Display current session info with ML capabilities
    current_memory = st.session_state.agent.get_or_create_memory(st.session_state.session_id)
    if current_memory.turns:
        ml_queries = len([t for t in current_memory.turns if t.query_type in [
            QueryType.CHURN_PREDICTION, QueryType.RECOMMENDATION_SYSTEM, 
            QueryType.CUSTOMER_SEGMENTATION, QueryType.LIFETIME_VALUE
        ]])
        analytics_info = f" | ML Queries: {ml_queries} | Mode: {st.session_state.analytics_settings['mode']}"
        st.caption(f"Session: {st.session_state.session_id} | Questions: {len(current_memory.turns)} | Started: {current_memory.turns[0].timestamp.strftime('%Y-%m-%d %H:%M')}{analytics_info}")
    else:
        st.caption(f"Session: {st.session_state.session_id} | New session | Mode: {st.session_state.analytics_settings['mode']}")
    
    # Enhanced query input with ML examples
    st.subheader("🎯 Ask Your Question")
    
    # ML-focused example queries
    example_queries = [
        "Predict which customers are likely to churn",
        "Generate personalized recommendations for cross-selling",
        "Segment customers by value and behavior",
        "Calculate customer lifetime value and growth potential",
        "Forecast occupancy trends for next quarter",
        "What strategic opportunities should we pursue?",
        "Analyze payment patterns and predict collection risks",
        "Recommend retention strategies for at-risk customers"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_value = ""
        if 'suggested_query' in st.session_state:
            default_value = st.session_state.suggested_query
            del st.session_state.suggested_query
        
        query = st.text_area(
            "Your question:",
            placeholder="e.g., 'Which customers are at risk of churning?' or 'Generate cross-sell recommendations'",
            height=100,
            value=default_value,
            key=f"current_query_{st.session_state.session_id}"
        )
    
    with col2:
        st.write("**ML Example Queries:**")
        for i, example in enumerate(example_queries[:4]):
            if st.button(f"🤖 {example[:30]}...", key=f"ml_example_{i}", use_container_width=True):
                st.session_state.suggested_query = example
                st.rerun()
    
    # Process query with enhanced ML display
    if st.button("🚀 Ask Question", type="primary", use_container_width=True) and query:
        with st.spinner("🤖 Processing your question with ML analytics..."):
            result = st.session_state.agent.process_query(
                query, 
                db_path, 
                st.session_state.session_id
            )
        
        # Add to conversation history
        timestamp = datetime.now()
        ai_response = result.get('response', 'Processed successfully')
        st.session_state.conversation_history.append((query, ai_response, timestamp))
        
        # Enhanced display for ML results
        if result["success"]:
            
            # Enhanced display for churn prediction
            if result["type"] == "churn_prediction":
                st.subheader("🚨 Customer Churn Prediction")
                
                # Show churn analysis summary
                if result.get("churn_analysis"):
                    churn_data = result["churn_analysis"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Customers", churn_data["total_customers_analyzed"])
                    with col2:
                        st.metric("High-Risk Customers", churn_data["high_risk_customers"])
                    with col3:
                        st.metric("Avg Churn Probability", f"{churn_data['average_churn_probability']:.1%}")
                    with col4:
                        st.metric("Predicted Monthly Churn", churn_data["predicted_monthly_churn"])
                
                # Show results table
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("🎯 High-Risk Customers")
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Churn Analysis",
                        csv,
                        f"churn_analysis_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show main response
                st.markdown(result["response"])
            
            # Enhanced display for recommendation system
            elif result["type"] == "recommendation_system":
                st.subheader("🎯 Recommendation Engine Results")
                
                # Show recommendation summary
                if result.get("recommendation_analysis"):
                    rec_data = result["recommendation_analysis"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Customers Analyzed", rec_data["total_customers_analyzed"])
                    with col2:
                        st.metric("High-Potential Customers", rec_data["high_potential_customers"])
                    with col3:
                        st.metric("Revenue Opportunity", f"${rec_data['revenue_optimization_opportunity']:,.2f}")
                
                # Show results table
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("💰 Top Opportunities")
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Recommendations",
                        csv,
                        f"recommendations_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show main response
                st.markdown(result["response"])
            
            # Enhanced display for customer segmentation
            elif result["type"] == "customer_segmentation":
                st.subheader("👥 Customer Segmentation Analysis")
                
                # Show segment overview
                if result.get("segment_stats"):
                    segment_data = result["segment_stats"]
                    total_customers = sum([stats['count'] for stats in segment_data.values()])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", total_customers)
                    with col2:
                        st.metric("Segments Identified", len(segment_data))
                    with col3:
                        total_value = sum([stats['total_ltv'] for stats in segment_data.values()])
                        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
                
                # Show results table
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("📊 Segment Breakdown")
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download Segmentation",
                        csv,
                        f"segmentation_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show main response
                st.markdown(result["response"])
            
            # Enhanced display for lifetime value analysis
            elif result["type"] == "lifetime_value":
                st.subheader("💰 Customer Lifetime Value Analysis")
                
                # Show LTV summary
                if result.get("customer_insights"):
                    insights = result["customer_insights"]
                    total_ltv = sum([c.lifetime_value for c in insights])
                    avg_ltv = total_ltv / len(insights) if insights else 0
                    predicted_total = sum([c.predicted_ltv for c in insights])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Portfolio LTV", f"${total_ltv:,.2f}")
                    with col2:
                        st.metric("Average Customer LTV", f"${avg_ltv:,.2f}")
                    with col3:
                        st.metric("Predicted Total LTV", f"${predicted_total:,.2f}")
                    with col4:
                        growth = ((predicted_total - total_ltv) / total_ltv * 100) if total_ltv > 0 else 0
                        st.metric("Growth Potential", f"{growth:+.1f}%")
                
                # Show results table
                if result.get("results") is not None and not result["results"].empty:
                    st.subheader("🌟 Top Value Customers")
                    st.dataframe(result["results"], use_container_width=True)
                    
                    # Download option
                    csv = result["results"].to_csv(index=False)
                    st.download_button(
                        "📥 Download LTV Analysis",
                        csv,
                        f"ltv_analysis_{timestamp.strftime('%H%M%S')}.csv",
                        "text/csv"
                    )
                
                # Show main response
                st.markdown(result["response"])
            
            # General query display
            else:
                st.subheader("💬 AI Response")
                st.markdown(result["response"])
            
            # Show follow-up suggestions for ALL successful results
            if result.get("follow_up_suggestions"):
                st.subheader("🎯 Suggested Follow-up Questions")
                
                # Enhanced suggestions with ML focus
                suggestion_categories = {
                    "🤖 ML Analytics": [s for s in result["follow_up_suggestions"] if any(word in s.lower() for word in ['predict', 'recommend', 'segment', 'churn', 'ltv'])],
                    "🎯 Strategic": [s for s in result["follow_up_suggestions"] if any(word in s.lower() for word in ['strategic', 'strategy', 'opportunity', 'growth'])],
                    "📊 Analytical": [s for s in result["follow_up_suggestions"] if s not in [s for s in result["follow_up_suggestions"] if any(word in s.lower() for word in ['predict', 'recommend', 'segment', 'churn', 'ltv', 'strategic', 'strategy', 'opportunity', 'growth'])]]
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
    
    # Quick access to ML analytics
    st.divider()
    st.subheader("🤖 Quick ML Analytics")
    
    quick_cols = st.columns(4)
    
    quick_queries = [
        ("🚨 Churn Risk Analysis", "Predict which customers are at risk of churning and provide retention strategies"),
        ("🎯 Cross-sell Opportunities", "Generate personalized recommendations for cross-selling and upselling"),
        ("👥 Customer Segmentation", "Analyze customer segments and provide strategic insights for each group"),
        ("💰 Lifetime Value Analysis", "Calculate customer lifetime value and identify growth opportunities")
    ]
    
    for i, (title, query_text) in enumerate(quick_queries):
        with quick_cols[i]:
            if st.button(title, key=f"ml_quick_{i}", use_container_width=True):
                st.session_state.suggested_query = query_text
                st.rerun()

if __name__ == "__main__":
    main()
