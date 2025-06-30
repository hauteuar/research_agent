# agents/chat_agent.py
"""
Opulence Chat Agent - Intelligent conversational interface for mainframe analysis
Provides natural language interaction with the analyzed codebase
"""

import asyncio
import logging
import json
import sqlite3
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from vllm import AsyncLLMEngine, SamplingParams
import uuid


@dataclass
class ChatContext:
    """Context information for chat conversations"""
    conversation_id: str
    user_query: str
    conversation_history: List[Dict[str, str]]
    relevant_components: List[str]
    analysis_results: Dict[str, Any]
    chat_type: str  # 'general', 'analysis', 'search', 'comparison'


class OpulenceChatAgent:
    """Enhanced chat agent for natural language interaction with mainframe analysis"""
    
    def __init__(self, coordinator, llm_engine: Optional[AsyncLLMEngine] = None, 
                 db_path: str = "opulence_data.db", gpu_id: int = 0):
        self.coordinator = coordinator
        self.llm_engine = llm_engine
        self.db_path = db_path
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(__name__)
        
        # Chat patterns for different types of queries
        self.query_patterns = {
            'analysis': [
                r'analyze|analysis|examine|investigate|study',
                r'what (is|are|does)|how (does|is|are)',
                r'explain|describe|tell me about',
                r'show me|find|search for'
            ],
            'lineage': [
                r'trace|lineage|flow|path|journey',
                r'where (is|does|comes|goes)',
                r'source|destination|origin',
                r'upstream|downstream|dependencies'
            ],
            'comparison': [
                r'compare|difference|similar|contrast',
                r'versus|vs|against|between',
                r'same|different|alike'
            ],
            'impact': [
                r'impact|effect|affect|change',
                r'what happens if|what if',
                r'consequences|result|outcome'
            ],
            'search': [
                r'find|search|look for|locate',
                r'contains|includes|has',
                r'pattern|like|similar to'
            ]
        }
        
        # Knowledge base for enhanced responses
        self.knowledge_base = {
            'cobol_concepts': {
                'working_storage': 'WORKING-STORAGE SECTION contains variables and data structures used by the program',
                'procedure_division': 'PROCEDURE DIVISION contains the executable code and business logic',
                'identification_division': 'IDENTIFICATION DIVISION identifies the program and provides metadata',
                'data_division': 'DATA DIVISION defines all data structures used in the program'
            },
            'jcl_concepts': {
                'job': 'JOB statement defines a unit of work to be processed by the system',
                'exec': 'EXEC statement specifies a program or procedure to be executed',
                'dd': 'DD (Data Definition) statements define input/output datasets'
            },
            'business_terms': {
                'settlement': 'Process of finalizing and recording financial transactions',
                'trade': 'Financial transaction involving buying or selling securities',
                'position': 'Current holdings of securities or cash',
                'reconciliation': 'Process of matching and verifying transaction records'
            }
        }
    
    async def process_chat_query(self, query: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Process a chat query and return intelligent response"""
        try:
            # Create chat context
            context = ChatContext(
                conversation_id=str(uuid.uuid4()),
                user_query=query,
                conversation_history=conversation_history or [],
                relevant_components=[],
                analysis_results={},
                chat_type=self._classify_query_type(query)
            )
            
            # Extract components mentioned in query
            context.relevant_components = self._extract_components_from_query(query)
            
            # Get relevant analysis results if components found
            if context.relevant_components:
                context.analysis_results = await self._get_relevant_analysis(context.relevant_components)
            
            # Generate response based on query type
            if context.chat_type == 'analysis':
                return await self._handle_analysis_query(context)
            elif context.chat_type == 'lineage':
                return await self._handle_lineage_query(context)
            elif context.chat_type == 'comparison':
                return await self._handle_comparison_query(context)
            elif context.chat_type == 'search':
                return await self._handle_search_query(context)
            elif context.chat_type == 'impact':
                return await self._handle_impact_query(context)
            else:
                return await self._handle_general_query(context)
                
        except Exception as e:
            self.logger.error(f"Chat query processing failed: {str(e)}")
            return {
                "response": f"I encountered an error processing your query: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try rephrasing your question", "Check if the component name is correct"]
            }
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query based on patterns"""
        query_lower = query.lower()
        
        # Score each query type
        scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        
        # Return the highest scoring type, default to 'general'
        if scores and max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'
    
    def _extract_components_from_query(self, query: str) -> List[str]:
        """Extract component names from the query"""
        components = []
        
        # Look for uppercase words (typical mainframe naming)
        uppercase_words = re.findall(r'\b[A-Z][A-Z0-9_-]{2,}\b', query)
        components.extend(uppercase_words)
        
        # Look for quoted components
        quoted_components = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
        components.extend(quoted_components)
        
        # Look for common patterns like "program XXXX" or "file YYYY"
        pattern_matches = re.findall(r'(?:program|file|table|field|job)\s+([A-Za-z0-9_-]+)', query, re.IGNORECASE)
        components.extend(pattern_matches)
        
        return list(set(components))  # Remove duplicates
    
    async def _get_relevant_analysis(self, components: List[str]) -> Dict[str, Any]:
        """Get analysis results for relevant components"""
        analysis_results = {}
        
        for component in components:
            try:
                # Try to get existing analysis from coordinator
                result = await self.coordinator.analyze_component(component, "auto-detect")
                if result and result.get("status") != "error":
                    analysis_results[component] = result
            except Exception as e:
                self.logger.warning(f"Could not analyze component {component}: {e}")
        
        return analysis_results
    
    async def _handle_analysis_query(self, context: ChatContext) -> Dict[str, Any]:
        """Handle analysis-type queries"""
        if not context.relevant_components:
            return {
                "response": "I'd be happy to analyze a component for you! Please specify which program, file, table, or field you'd like me to analyze.",
                "response_type": "clarification",
                "suggestions": [
                    "Try: 'Analyze program TRADE_PROC'",
                    "Try: 'What does CUSTOMER_FILE contain?'",
                    "Try: 'Explain the ACCOUNT_ID field'"
                ]
            }
        
        component = context.relevant_components[0]
        analysis = context.analysis_results.get(component)
        
        if not analysis:
            return {
                "response": f"I couldn't find '{component}' in the analyzed codebase. Make sure the component name is correct and the files have been processed.",
                "response_type": "not_found",
                "suggestions": [
                    "Check the spelling of the component name",
                    "Ensure the files containing this component have been uploaded",
                    "Try a partial name search"
                ]
            }
        
        # Generate comprehensive analysis response
        response = await self._generate_analysis_response(component, analysis, context.user_query)
        
        return {
            "response": response,
            "response_type": "analysis",
            "component": component,
            "analysis_data": analysis,
            "suggestions": self._generate_follow_up_suggestions(component, analysis)
        }
    
    async def _handle_lineage_query(self, context: ChatContext) -> Dict[str, Any]:
        """Handle lineage-type queries - FIXED"""
        if not context.relevant_components:
            return {
                "response": "To trace lineage, please specify a field, file, or data element you'd like me to trace.",
                "response_type": "clarification",
                "suggestions": [
                    "Try: 'Trace the lineage of TRADE_DATE'",
                    "Try: 'Where does CUSTOMER_ID come from?'",
                    "Try: 'Show the flow of ACCOUNT_BALANCE'"
                ]
            }
        
        component = context.relevant_components[0]
        
        # Perform lineage analysis - FIXED
        try:
            # FIX: Use single GPU coordinator approach
            lineage_agent = self.coordinator.get_agent("lineage_analyzer")
            lineage_result = await lineage_agent.analyze_field_lineage(component)
            
            response = await self._generate_lineage_response(component, lineage_result, context.user_query)
            
            return {
                "response": response,
                "response_type": "lineage",
                "component": component,
                "lineage_data": lineage_result,
                "suggestions": [
                    f"Show impact analysis for {component}",
                    f"Find programs that modify {component}",
                    f"Trace {component} dependencies"
                ]
            }
        except Exception as e:
            return {
                "response": f"I couldn't trace the lineage for '{component}': {str(e)}",
                "response_type": "error",
                "suggestions": ["Try a different component name", "Check if the component exists in the processed files"]
            }

    
    async def _handle_search_query(self, context: ChatContext) -> Dict[str, Any]:
        """Handle search-type queries - FIXED"""
        # Extract search terms
        search_terms = self._extract_search_terms(context.user_query)
        
        if not search_terms:
            return {
                "response": "What would you like me to search for? I can find programs, fields, business logic, or code patterns.",
                "response_type": "clarification",
                "suggestions": [
                    "Try: 'Find programs that calculate interest'",
                    "Try: 'Search for validation logic'",
                    "Try: 'Show me programs using CUSTOMER_ID'"
                ]
            }
        
        # Perform vector search - FIXED
        try:
            # FIX: Use single GPU coordinator approach
            vector_agent = self.coordinator.get_agent("vector_index")
            search_results = await vector_agent.semantic_search(" ".join(search_terms), top_k=10)
            
            response = await self._generate_search_response(search_terms, search_results, context.user_query)
            
            return {
                "response": response,
                "response_type": "search",
                "search_terms": search_terms,
                "search_results": search_results,
                "suggestions": [
                    "Refine search with more specific terms",
                    "Analyze one of the found components",
                    "Search for related patterns"
                ]
            }
        except Exception as e:
            return {
                "response": f"Search failed: {str(e)}",
                "response_type": "error",
                "suggestions": ["Try different search terms", "Check if files have been processed"]
            }
    
    async def _handle_comparison_query(self, context: ChatContext) -> Dict[str, Any]:
        """Handle comparison-type queries"""
        if len(context.relevant_components) < 2:
            return {
                "response": "To compare components, please specify two items you'd like me to compare.",
                "response_type": "clarification",
                "suggestions": [
                    "Try: 'Compare PROGRAM_A and PROGRAM_B'",
                    "Try: 'What's the difference between OLD_FILE and NEW_FILE?'",
                    "Try: 'How are FIELD_X and FIELD_Y different?'"
                ]
            }
        
        comp1, comp2 = context.relevant_components[:2]
        
        # Get analysis for both components
        analysis1 = context.analysis_results.get(comp1)
        analysis2 = context.analysis_results.get(comp2)
        
        response = await self._generate_comparison_response(comp1, comp2, analysis1, analysis2, context.user_query)
        
        return {
            "response": response,
            "response_type": "comparison",
            "components": [comp1, comp2],
            "analysis_data": {comp1: analysis1, comp2: analysis2},
            "suggestions": [
                f"Analyze {comp1} in detail",
                f"Analyze {comp2} in detail",
                "Find similar components"
            ]
        }
    
    async def _handle_impact_query(self, context: ChatContext) -> Dict[str, Any]:
        """Handle impact analysis queries"""
        if not context.relevant_components:
            return {
                "response": "To analyze impact, please specify which component you're considering changing.",
                "response_type": "clarification",
                "suggestions": [
                    "Try: 'What's the impact of changing INTEREST_RATE?'",
                    "Try: 'What happens if I modify CUSTOMER_PROC?'",
                    "Try: 'Show dependencies for TRADE_FILE'"
                ]
            }
        
        component = context.relevant_components[0]
        analysis = context.analysis_results.get(component)
        
        # Generate impact analysis
        response = await self._generate_impact_response(component, analysis, context.user_query)
        
        return {
            "response": response,
            "response_type": "impact",
            "component": component,
            "analysis_data": analysis,
            "suggestions": [
                f"Show detailed lineage for {component}",
                f"Find all programs using {component}",
                "Analyze testing requirements"
            ]
        }
    
    async def _handle_general_query(self, context: ChatContext) -> Dict[str, Any]:
        """Handle general queries and provide helpful guidance"""
        
        # Check if user is asking about capabilities
        if any(word in context.user_query.lower() for word in ['help', 'can you', 'what can', 'how to']):
            return {
                "response": self._generate_help_response(),
                "response_type": "help",
                "suggestions": [
                    "Analyze a specific component",
                    "Search for code patterns",
                    "Trace field lineage"
                ]
            }
        
        # Generate contextual response using LLM
        response = await self._generate_contextual_response(context)
        
        return {
            "response": response,
            "response_type": "general",
            "suggestions": [
                "Ask about a specific program or file",
                "Search for business logic",
                "Analyze component relationships"
            ]
        }
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """Extract search terms from query"""
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        words = re.findall(r'\b\w+\b', query.lower())
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _generate_follow_up_suggestions(self, component: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate follow-up suggestions based on analysis"""
        suggestions = []
        
        if analysis and isinstance(analysis, dict):
            # Component-specific suggestions
            component_type = analysis.get("component_type", "")
            
            if component_type == "program":
                suggestions.extend([
                    f"Show business logic in {component}",
                    f"Find programs similar to {component}",
                    f"Analyze complexity of {component}"
                ])
            elif component_type == "field":
                suggestions.extend([
                    f"Trace {component} data flow",
                    f"Find all programs using {component}",
                    f"Show impact of changing {component}"
                ])
            elif component_type == "file":
                suggestions.extend([
                    f"Show programs reading {component}",
                    f"Analyze {component} structure",
                    f"Compare {component} with similar files"
                ])
            
            # Analysis-specific suggestions
            if analysis.get("logic_analysis"):
                suggestions.append(f"Explain business rules in {component}")
            
            if analysis.get("lineage"):
                suggestions.append(f"Show full lineage map for {component}")
        
        # Default suggestions if none generated
        if not suggestions:
            suggestions = [
                f"Tell me more about {component}",
                f"Find components similar to {component}",
                f"Show {component} dependencies"
            ]
        
        return suggestions[:3]  # Return top 3
    
    async def _generate_analysis_response(self, component: str, analysis: Dict, user_query: str) -> str:
        """Generate intelligent analysis response using LLM with retrieved context - FIXED"""
        await self._ensure_llm_engine()
        
        # Prepare context from analysis results
        context_parts = [f"Component: {component}"]
        
        if analysis.get("basic_info"):
            basic_info = analysis["basic_info"]
            if isinstance(basic_info, dict):
                context_parts.append(f"Type: {analysis.get('component_type', 'unknown')}")
                context_parts.append(f"Chunks found: {basic_info.get('total_chunks', 0)}")
                if basic_info.get("chunk_summary"):
                    chunk_types = ", ".join(basic_info["chunk_summary"].keys())
                    context_parts.append(f"Code structure: {chunk_types}")
        
        if analysis.get("logic_analysis"):
            logic_data = analysis["logic_analysis"]
            if isinstance(logic_data, dict) and "error" not in logic_data:
                if "complexity_score" in logic_data:
                    context_parts.append(f"Complexity score: {logic_data['complexity_score']:.2f}")
                if "business_rules" in logic_data:
                    rules_count = len(logic_data.get("business_rules", []))
                    context_parts.append(f"Business rules found: {rules_count}")
                if "recommendations" in logic_data:
                    context_parts.append("Recommendations available")
        
        if analysis.get("lineage"):
            lineage_data = analysis["lineage"]
            if isinstance(lineage_data, dict) and "error" not in lineage_data:
                if "usage_analysis" in lineage_data:
                    usage_stats = lineage_data["usage_analysis"].get("statistics", {})
                    total_refs = usage_stats.get("total_references", 0)
                    context_parts.append(f"Total references: {total_refs}")
        
        # Get semantic search results for additional context - FIXED
        search_context = ""
        try:
            # FIX: Use single GPU coordinator approach
            vector_agent = self.coordinator.get_agent("vector_index")
            search_results = await vector_agent.semantic_search(f"{component} {user_query}", top_k=3)
            if search_results:
                search_context = "\n\nRelated code patterns found:\n"
                for i, result in enumerate(search_results[:2], 1):
                    metadata = result.get('metadata', {})
                    search_context += f"{i}. {metadata.get('program_name', 'Unknown')} ({metadata.get('chunk_type', 'code')})\n"
                    search_context += f"   Content: {result.get('content', '')[:150]}...\n"
        except Exception as e:
            self.logger.warning(f"Could not get search context: {e}")
        
        context = "\n".join(context_parts) + search_context
        
        prompt = f"""
        You are Opulence, an expert mainframe analysis assistant. Answer the user's question about the analyzed component using the provided context.

        User Question: "{user_query}"

        Analysis Context:
        {context}

        Instructions:
        - Provide a clear, informative response about the component
        - Use technical details from the analysis context
        - If the component has business logic, explain its purpose
        - If there are recommendations or risks, mention them
        - Be conversational but technically accurate
        - Keep the response concise but comprehensive

        Response:
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=400, stop=["\n\nUser:", "Question:"])
        
        try:
            response = await self._generate_with_llm(prompt, sampling_params)
            return response.strip()
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {str(e)}")
            return f"I found information about {component}, but encountered an error generating the response. The analysis shows {len(context_parts)} key aspects of this component."

    async def _generate_lineage_response(self, component: str, lineage_result: Dict, user_query: str) -> str:
        """Generate intelligent lineage response using LLM"""
        await self._ensure_llm_engine()
        
        # Prepare lineage context
        context_parts = [f"Component: {component}"]
        
        if isinstance(lineage_result, dict) and "error" not in lineage_result:
            if "usage_analysis" in lineage_result:
                usage = lineage_result["usage_analysis"]
                stats = usage.get("statistics", {})
                context_parts.append(f"Total references: {stats.get('total_references', 0)}")
                context_parts.append(f"Programs using it: {len(stats.get('programs_using', []))}")
                
                if stats.get("programs_using"):
                    programs = ", ".join(stats["programs_using"][:5])
                    context_parts.append(f"Used by programs: {programs}")
            
            if "lineage_graph" in lineage_result:
                graph = lineage_result["lineage_graph"]
                if isinstance(graph, dict):
                    nodes = graph.get("total_nodes", 0)
                    edges = graph.get("total_edges", 0)
                    context_parts.append(f"Lineage network: {nodes} components, {edges} connections")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
        You are Opulence, an expert at tracing data lineage in mainframe systems. Answer the user's lineage question using the analysis results.

        User Question: "{user_query}"

        Lineage Analysis:
        {context}

        Instructions:
        - Explain the data flow and usage patterns for this component
        - Describe which programs use this component and how
        - If there are multiple references, explain the relationships
        - Mention any potential impact of changes
        - Be specific about the lineage findings

        Response:
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=350, stop=["\n\nUser:", "Question:"])
        
        try:
            response = await self._generate_with_llm(prompt, sampling_params)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Lineage response generation failed: {str(e)}")
            return f"I traced the lineage for {component} and found {context.count('programs_using') > 0 and 'multiple' or 'some'} references in the codebase."
    
    async def _generate_search_response(self, search_terms: List[str], search_results: List[Dict], user_query: str) -> str:
        """Generate intelligent search response using LLM"""
        await self._ensure_llm_engine()
        
        if not search_results:
            return f"I couldn't find any code patterns matching '{' '.join(search_terms)}'. Try different search terms or check if the relevant files have been processed."
        
        # Prepare search context
        context_parts = [f"Search terms: {', '.join(search_terms)}"]
        context_parts.append(f"Found {len(search_results)} matches")
        
        # Add top results to context
        for i, result in enumerate(search_results[:3], 1):
            metadata = result.get('metadata', {})
            similarity = result.get('similarity_score', 0)
            context_parts.append(f"Result {i}: {metadata.get('program_name', 'Unknown')} (similarity: {similarity:.2f})")
            context_parts.append(f"  Type: {metadata.get('chunk_type', 'code')}")
            context_parts.append(f"  Content: {result.get('content', '')[:200]}...")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
        You are Opulence, helping users find relevant code in mainframe systems. Present the search results in a helpful way.

        User Query: "{user_query}"

        Search Results:
        {context}

        Instructions:
        - Summarize what was found based on the search terms
        - Highlight the most relevant matches
        - Explain what each result contains and why it's relevant
        - Suggest follow-up actions if appropriate
        - Be specific about the code patterns found

        Response:
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=400, stop=["\n\nUser:", "Query:"])
        
        try:
            response = await self._generate_with_llm(prompt, sampling_params)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Search response generation failed: {str(e)}")
            return f"Found {len(search_results)} code patterns matching your search. The most relevant result is from {search_results[0].get('metadata', {}).get('program_name', 'a program')}."
    
    async def _generate_comparison_response(self, comp1: str, comp2: str, analysis1: Dict, analysis2: Dict, user_query: str) -> str:
        """Generate intelligent comparison response using LLM"""
        await self._ensure_llm_engine()
        
        # Prepare comparison context
        context_parts = [f"Comparing: {comp1} vs {comp2}"]
        
        # Add analysis data for both components
        for comp_name, analysis in [(comp1, analysis1), (comp2, analysis2)]:
            if analysis and isinstance(analysis, dict) and "error" not in analysis:
                context_parts.append(f"\n{comp_name}:")
                context_parts.append(f"  Type: {analysis.get('component_type', 'unknown')}")
                context_parts.append(f"  Chunks: {analysis.get('chunks_found', 0)}")
                
                if analysis.get("logic_analysis"):
                    logic = analysis["logic_analysis"]
                    if isinstance(logic, dict) and "error" not in logic:
                        if "complexity_score" in logic:
                            context_parts.append(f"  Complexity: {logic['complexity_score']:.2f}")
                        if "business_rules" in logic:
                            context_parts.append(f"  Business rules: {len(logic.get('business_rules', []))}")
            else:
                context_parts.append(f"\n{comp_name}: No detailed analysis available")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
        You are Opulence, comparing mainframe components. Provide a detailed comparison based on the analysis data.

        User Question: "{user_query}"

        Comparison Data:
        {context}

        Instructions:
        - Compare the two components across relevant dimensions
        - Highlight key similarities and differences
        - Discuss complexity, functionality, and structure differences
        - Suggest which component might be more suitable for different purposes
        - Be specific about what makes them different or similar

        Response:
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=450, stop=["\n\nUser:", "Question:"])
        
        try:
            response = await self._generate_with_llm(prompt, sampling_params)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Comparison response generation failed: {str(e)}")
            return f"I compared {comp1} and {comp2}. Both components have different characteristics based on the available analysis data."
    
    async def _generate_impact_response(self, component: str, analysis: Dict, user_query: str) -> str:
        """Generate intelligent impact analysis response using LLM"""
        await self._ensure_llm_engine()
        
        # Prepare impact context
        context_parts = [f"Component: {component}"]
        
        if analysis and isinstance(analysis, dict) and "error" not in analysis:
            # Add complexity and risk factors
            if analysis.get("logic_analysis"):
                logic = analysis["logic_analysis"]
                if isinstance(logic, dict) and "error" not in logic:
                    if "complexity_score" in logic:
                        complexity = logic["complexity_score"]
                        context_parts.append(f"Complexity score: {complexity:.2f}")
                        if complexity > 7:
                            context_parts.append("Risk level: HIGH - Complex code with many dependencies")
                        elif complexity > 4:
                            context_parts.append("Risk level: MEDIUM - Moderate complexity")
                        else:
                            context_parts.append("Risk level: LOW - Simple, straightforward code")
                    
                    if "recommendations" in logic and logic["recommendations"]:
                        context_parts.append("Has specific recommendations for changes")
            
            # Add usage information
            if analysis.get("lineage"):
                lineage = analysis["lineage"]
                if isinstance(lineage, dict) and "error" not in lineage:
                    if "usage_analysis" in lineage:
                        usage = lineage["usage_analysis"].get("statistics", {})
                        total_refs = usage.get("total_references", 0)
                        programs_using = len(usage.get("programs_using", []))
                        context_parts.append(f"Used by {programs_using} programs with {total_refs} total references")
        
        context = "\n".join(context_parts)
        
        prompt = f"""
        You are Opulence, analyzing the impact of potential changes to mainframe components. Assess the risks and implications.

        User Question: "{user_query}"

        Impact Analysis Context:
        {context}

        Instructions:
        - Assess the potential impact of changing this component
        - Consider complexity, usage patterns, and dependencies
        - Provide specific recommendations for managing the change
        - Discuss testing requirements and rollback considerations
        - Mention any programs or systems that could be affected

        Response:
        """
        
        sampling_params = SamplingParams(temperature=0.3, max_tokens=400, stop=["\n\nUser:", "Question:"])
        
        try:
            response = await self._generate_with_llm(prompt, sampling_params)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Impact response generation failed: {str(e)}")
            return f"Changes to {component} could have significant impact based on its complexity and usage patterns. Thorough testing would be recommended."
    
    async def _generate_contextual_response(self, context: ChatContext) -> str:
        """Generate contextual response for general queries using LLM"""
        await self._ensure_llm_engine()
        
        # Build context from conversation history and available data
        context_parts = [f"User query: {context.user_query}"]
        
        if context.conversation_history:
            recent_history = context.conversation_history[-3:]  # Last 3 exchanges
            context_parts.append("Recent conversation:")
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                context_parts.append(f"  {role}: {content}...")
        
        if context.relevant_components:
            context_parts.append(f"Mentioned components: {', '.join(context.relevant_components)}")
        
        # Get system status for context
        try:
            if hasattr(self.coordinator, 'get_statistics'):
                stats = await self.coordinator.get_statistics()
                system_stats = stats.get("system_stats", {})
                files_processed = system_stats.get("total_files_processed", 0)
                context_parts.append(f"System status: {files_processed} files processed")
        except:
            pass
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""
        You are Opulence, a friendly and knowledgeable mainframe analysis assistant. Respond to the user's query helpfully and conversationally.

        Context:
        {context_str}

        Your capabilities include:
        - Analyzing COBOL programs, JCL jobs, and data files
        - Tracing field lineage and data flow
        - Finding code patterns and business logic
        - Comparing components and assessing impact
        - Generating technical documentation

        Instructions:
        - Be helpful and engaging
        - If the user needs specific analysis, guide them on how to request it
        - Provide relevant examples based on mainframe development
        - Keep responses concise but informative
        - Use your knowledge of mainframe systems appropriately

        Response:
        """
        
        sampling_params = SamplingParams(temperature=0.4, max_tokens=300, stop=["\n\nUser:", "Context:"])
        
        try:
            response = await self._generate_with_llm(prompt, sampling_params)
            return response.strip()
        except Exception as e:
            self.logger.error(f"Contextual response generation failed: {str(e)}")
            return self._generate_help_response()
    
    def _generate_help_response(self) -> str:
        """Generate help response"""
        return """
I'm Opulence, your mainframe analysis assistant! I can help you understand and analyze your COBOL programs, JCL jobs, and data files.

**What I can do:**
🔍 **Analyze Components** - Deep dive into programs, files, tables, or fields
📊 **Trace Lineage** - Follow data flow and dependencies
🔍 **Search Code** - Find patterns, business logic, or specific functionality  
📈 **Impact Analysis** - Assess the effects of potential changes
🔄 **Compare Components** - Side-by-side analysis of different elements

**Try asking:**
- "Analyze the CUSTOMER_PROC program"
- "Trace the lineage of ACCOUNT_BALANCE field"
- "Find all programs that calculate interest"
- "What's the impact of changing TRADE_DATE?"
- "Compare SETTLEMENT_OLD and SETTLEMENT_NEW"

Just describe what you'd like to know about your mainframe systems!
        """
    
    async def _ensure_llm_engine(self):
        """Ensure LLM engine is available - reuse coordinator's engines"""
        if self.llm_engine is not None:
            return
        
        # Try to get from coordinator first
        if self.coordinator is not None:
            try:
                best_gpu = await self.coordinator.get_available_gpu_for_agent("chat_agent")
                if best_gpu is not None:
                    self.llm_engine = await self.coordinator.get_or_create_llm_engine(best_gpu)
                    self.gpu_id = best_gpu
                    self.logger.info(f"Chat agent using coordinator's LLM on GPU {best_gpu}")
                    return
            except Exception as e:
                self.logger.warning(f"Failed to get LLM from coordinator: {e}")
        
        # Fallback: try global coordinator
        try:
            from opulence_coordinator import get_dynamic_coordinator
            global_coordinator = get_dynamic_coordinator()
            best_gpu = await global_coordinator.get_available_gpu_for_agent("chat_agent")
            if best_gpu is not None:
                self.llm_engine = await global_coordinator.get_or_create_llm_engine(best_gpu)
                self.gpu_id = best_gpu
                self.logger.info(f"Chat agent using global coordinator's LLM on GPU {best_gpu}")
        except Exception as e:
            self.logger.error(f"Failed to get LLM engine: {e}")
            raise
    
    async def _generate_with_llm(self, prompt: str, sampling_params) -> str:
        """Generate text with LLM - handles both old and new vLLM API"""
        try:
            # Try new API first (with request_id)
            request_id = str(uuid.uuid4())
            async_generator = self.llm_engine.generate(prompt, sampling_params, request_id=request_id)
            
            # Handle the async generator
            async for result in async_generator:
                return result.outputs[0].text.strip()
                
        except TypeError as e:
            if "request_id" in str(e):
                # Fallback to old API (without request_id)
                async_generator = self.llm_engine.generate(prompt, sampling_params)
                async for result in async_generator:
                    return result.outputs[0].text.strip()
            else:
                raise e
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            return ""