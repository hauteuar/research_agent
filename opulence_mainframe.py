"""
LLM-Powered Intelligent Mainframe Analysis System
Integrates CodeLlama and intelligent analysis with file vs DB2 comparison
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
from datetime import datetime

# LLM Integration (using vLLM for CodeLlama)
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️ vLLM not available - falling back to pattern-based analysis")

@dataclass
class IntelligentAnalysis:
    """Results from LLM-powered intelligent analysis"""
    component_name: str
    component_type: str
    
    # LLM-generated insights
    business_purpose: str = ""
    data_flow_description: str = ""
    complexity_assessment: str = ""
    modernization_recommendations: List[str] = field(default_factory=list)
    
    # File-DB2 integration insights
    data_consistency_issues: List[str] = field(default_factory=list)
    schema_migration_plan: List[str] = field(default_factory=list)
    potential_data_quality_problems: List[str] = field(default_factory=list)
    
    # Cross-component relationships
    upstream_dependencies: List[str] = field(default_factory=list)
    downstream_impacts: List[str] = field(default_factory=list)
    shared_data_patterns: List[str] = field(default_factory=list)
    
    # Risk assessment
    change_risk_level: str = "MEDIUM"
    risk_factors: List[str] = field(default_factory=list)
    mitigation_strategies: List[str] = field(default_factory=list)
    
    # Confidence scores
    analysis_confidence: float = 0.0
    llm_reasoning: str = ""

class LLMAnalysisEngine:
    """LLM-powered analysis engine using CodeLlama"""
    
    def __init__(self, model_name: str = "codellama/CodeLlama-34b-Instruct-hf"):
        self.model_name = model_name
        self.llm = None
        self.tokenizer = None
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
            stop=["</analysis>", "Human:", "Assistant:"]
        )
        self.initialize_llm()
    
    def initialize_llm(self):
        """Initialize the LLM model"""
        if VLLM_AVAILABLE:
            try:
                self.llm = LLM(
                    model=self.model_name,
                    tensor_parallel_size=2,
                    max_model_len=16384,
                    gpu_memory_utilization=0.8
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                print(f"✅ Initialized LLM: {self.model_name}")
            except Exception as e:
                print(f"❌ Failed to initialize LLM: {e}")
                self.llm = None
        else:
            print("⚠️ LLM not available - using pattern-based analysis")
    
    def analyze_component_intelligence(self, component_data: Dict[str, Any], 
                                     related_components: List[Dict] = None,
                                     file_db2_comparison: Dict = None) -> IntelligentAnalysis:
        """Perform intelligent analysis of a mainframe component using LLM"""
        
        if not self.llm:
            return self._fallback_analysis(component_data)
        
        # Build comprehensive prompt
        prompt = self._build_analysis_prompt(component_data, related_components, file_db2_comparison)
        
        # Generate analysis using LLM
        outputs = self.llm.generate([prompt], self.sampling_params)
        llm_response = outputs[0].outputs[0].text
        
        # Parse LLM response into structured analysis
        analysis = self._parse_llm_response(llm_response, component_data['name'], component_data['component_type'])
        
        # Enhance with additional intelligence
        analysis = self._enhance_with_cross_component_analysis(analysis, related_components)
        analysis = self._integrate_file_db2_insights(analysis, file_db2_comparison)
        
        return analysis
    
    def _build_analysis_prompt(self, component_data: Dict[str, Any], 
                              related_components: List[Dict] = None,
                              file_db2_comparison: Dict = None) -> str:
        """Build comprehensive analysis prompt for LLM"""
        
        component_name = component_data['name']
        component_type = component_data['component_type']
        analysis_data = component_data.get('analysis', {})
        content = component_data.get('content', '')[:8000]  # Limit content size
        
        prompt = f"""<analysis>
You are an expert mainframe systems analyst. Analyze this {component_type} component and provide intelligent insights.

COMPONENT: {component_name}
TYPE: {component_type}
SUBTYPE: {component_data.get('subtype', 'Unknown')}

COMPONENT SOURCE CODE:
{content}

ANALYSIS DATA:
- File Operations: {len(analysis_data.get('file_operations', []))}
- Fields Defined: {len(analysis_data.get('fields_defined', []))}
- Business Functions: {analysis_data.get('business_functions', [])}
- Complexity Score: {analysis_data.get('complexity_score', 0)}
- Files Created: {list(analysis_data.get('creates_files', []))}
- Files Read: {list(analysis_data.get('reads_files', []))}
- Files Updated: {list(analysis_data.get('updates_files', []))}

"""
        
        # Add related components context
        if related_components:
            prompt += f"""
RELATED COMPONENTS:
{self._format_related_components(related_components)}
"""
        
        # Add file vs DB2 comparison context
        if file_db2_comparison:
            prompt += f"""
FILE vs DB2 COMPARISON RESULTS:
{self._format_file_db2_context(file_db2_comparison)}
"""
        
        prompt += """
ANALYZE AND PROVIDE:

1. BUSINESS PURPOSE (2-3 sentences):
   What is the primary business function of this component?

2. DATA FLOW DESCRIPTION (3-4 sentences):
   How does data flow through this component? What transformations occur?

3. COMPLEXITY ASSESSMENT (2-3 sentences):
   Assess the complexity and maintainability. What makes it complex?

4. MODERNIZATION RECOMMENDATIONS (3-5 bullet points):
   Specific recommendations for modernizing this component.

5. DATA CONSISTENCY ISSUES (if file/DB2 comparison available, 2-4 bullet points):
   Identify potential data quality or consistency problems.

6. SCHEMA MIGRATION PLAN (if applicable, 3-5 bullet points):
   Steps for migrating or aligning schemas between file and database.

7. CHANGE RISK ASSESSMENT:
   Risk Level: [LOW/MEDIUM/HIGH]
   Risk Factors: [2-3 key risks]
   Mitigation Strategies: [2-3 strategies]

8. CROSS-COMPONENT RELATIONSHIPS (if related components provided):
   How this component interacts with others and potential impact of changes.

Provide your analysis in JSON format:
{
  "business_purpose": "...",
  "data_flow_description": "...",
  "complexity_assessment": "...",
  "modernization_recommendations": ["...", "..."],
  "data_consistency_issues": ["...", "..."],
  "schema_migration_plan": ["...", "..."],
  "change_risk_level": "MEDIUM",
  "risk_factors": ["...", "..."],
  "mitigation_strategies": ["...", "..."],
  "upstream_dependencies": ["...", "..."],
  "downstream_impacts": ["...", "..."],
  "shared_data_patterns": ["...", "..."],
  "analysis_confidence": 0.85,
  "llm_reasoning": "Analysis based on code structure, data operations, and cross-component relationships..."
}
</analysis>"""
        
        return prompt
    
    def _format_related_components(self, related_components: List[Dict]) -> str:
        """Format related components for prompt context"""
        context = ""
        for comp in related_components[:5]:  # Limit to 5 components
            context += f"- {comp['name']} ({comp['component_type']}): "
            context += f"Files: {comp.get('analysis', {}).get('creates_files', [])} "
            context += f"Operations: {len(comp.get('analysis', {}).get('file_operations', []))}\n"
        return context
    
    def _format_file_db2_context(self, file_db2_comparison: Dict) -> str:
        """Format file vs DB2 comparison context for prompt"""
        context = ""
        
        for file_name, comparison in file_db2_comparison.items():
            if 'error' in comparison:
                continue
                
            schema_comp = comparison.get('schema_comparison')
            data_comp = comparison.get('data_comparison')
            
            if schema_comp:
                context += f"File: {file_name}\n"
                context += f"- Schema Compatibility: {schema_comp.compatibility_score:.1%}\n"
                context += f"- Missing in DB2: {schema_comp.file_only_columns}\n"
                context += f"- Type Mismatches: {len(schema_comp.type_mismatches)}\n"
                
                if data_comp:
                    context += f"- Data Consistency: {data_comp.data_consistency_score:.1%}\n"
                    context += f"- Mismatched Records: {len(data_comp.mismatched_records)}\n"
                context += "\n"
        
        return context
    
    def _parse_llm_response(self, llm_response: str, component_name: str, 
                          component_type: str) -> IntelligentAnalysis:
        """Parse LLM response into structured analysis"""
        
        analysis = IntelligentAnalysis(
            component_name=component_name,
            component_type=component_type
        )
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                
                # Map parsed data to analysis object
                analysis.business_purpose = parsed_data.get('business_purpose', '')
                analysis.data_flow_description = parsed_data.get('data_flow_description', '')
                analysis.complexity_assessment = parsed_data.get('complexity_assessment', '')
                analysis.modernization_recommendations = parsed_data.get('modernization_recommendations', [])
                analysis.data_consistency_issues = parsed_data.get('data_consistency_issues', [])
                analysis.schema_migration_plan = parsed_data.get('schema_migration_plan', [])
                analysis.change_risk_level = parsed_data.get('change_risk_level', 'MEDIUM')
                analysis.risk_factors = parsed_data.get('risk_factors', [])
                analysis.mitigation_strategies = parsed_data.get('mitigation_strategies', [])
                analysis.upstream_dependencies = parsed_data.get('upstream_dependencies', [])
                analysis.downstream_impacts = parsed_data.get('downstream_impacts', [])
                analysis.shared_data_patterns = parsed_data.get('shared_data_patterns', [])
                analysis.analysis_confidence = parsed_data.get('analysis_confidence', 0.8)
                analysis.llm_reasoning = parsed_data.get('llm_reasoning', '')
                
        except (json.JSONDecodeError, AttributeError) as e:
            # Fallback to text parsing if JSON parsing fails
            print(f"⚠️ JSON parsing failed, using text extraction: {e}")
            analysis = self._extract_from_text(llm_response, analysis)
            analysis.analysis_confidence = 0.6  # Lower confidence for text parsing
        
        return analysis
    
    def _extract_from_text(self, text: str, analysis: IntelligentAnalysis) -> IntelligentAnalysis:
        """Extract insights from text when JSON parsing fails"""
        
        # Extract business purpose
        purpose_match = re.search(r'(?:BUSINESS PURPOSE|business_purpose)[:\s]*([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n[A-Z]|$)', text, re.IGNORECASE)
        if purpose_match:
            analysis.business_purpose = purpose_match.group(1).strip()
        
        # Extract recommendations
        rec_pattern = r'(?:MODERNIZATION|RECOMMENDATIONS?)[:\s]*\n((?:[-•*]\s*[^\n]+\n?)+)'
        rec_match = re.search(rec_pattern, text, re.IGNORECASE)
        if rec_match:
            recommendations = re.findall(r'[-•*]\s*([^\n]+)', rec_match.group(1))
            analysis.modernization_recommendations = recommendations
        
        # Extract risk level
        risk_match = re.search(r'Risk Level[:\s]*(\w+)', text, re.IGNORECASE)
        if risk_match:
            analysis.change_risk_level = risk_match.group(1).upper()
        
        return analysis
    
    def _enhance_with_cross_component_analysis(self, analysis: IntelligentAnalysis, 
                                             related_components: List[Dict] = None) -> IntelligentAnalysis:
        """Enhance analysis with cross-component insights"""
        
        if not related_components:
            return analysis
        
        # Analyze shared files and dependencies
        component_files = set()
        for comp in related_components:
            comp_analysis = comp.get('analysis', {})
            component_files.update(comp_analysis.get('creates_files', []))
            component_files.update(comp_analysis.get('reads_files', []))
            component_files.update(comp_analysis.get('updates_files', []))
        
        # Find shared data patterns
        shared_files = []
        for comp in related_components:
            comp_analysis = comp.get('analysis', {})
            comp_files = set(comp_analysis.get('creates_files', []) + 
                           comp_analysis.get('reads_files', []) + 
                           comp_analysis.get('updates_files', []))
            
            if comp_files & component_files:
                shared_files.append(f"{comp['name']} shares files: {list(comp_files & component_files)}")
        
        analysis.shared_data_patterns = shared_files
        
        return analysis
    
    def _integrate_file_db2_insights(self, analysis: IntelligentAnalysis, 
                                   file_db2_comparison: Dict = None) -> IntelligentAnalysis:
        """Integrate file vs DB2 comparison insights"""
        
        if not file_db2_comparison:
            return analysis
        
        # Analyze comparison results and add intelligent insights
        for file_name, comparison in file_db2_comparison.items():
            if 'error' in comparison:
                continue
            
            schema_comp = comparison.get('schema_comparison')
            data_comp = comparison.get('data_comparison')
            
            if schema_comp:
                # Schema insights
                if schema_comp.compatibility_score < 0.8:
                    analysis.potential_data_quality_problems.append(
                        f"Low schema compatibility ({schema_comp.compatibility_score:.1%}) for {file_name}"
                    )
                
                if schema_comp.type_mismatches:
                    analysis.schema_migration_plan.append(
                        f"Fix data type mismatches in {file_name}: {[m['column_name'] for m in schema_comp.type_mismatches]}"
                    )
                
                if schema_comp.file_only_columns:
                    analysis.schema_migration_plan.append(
                        f"Add missing columns to DB2 for {file_name}: {schema_comp.file_only_columns}"
                    )
            
            if data_comp:
                # Data insights
                if data_comp.data_consistency_score < 0.9:
                    analysis.data_consistency_issues.append(
                        f"Data inconsistency in {file_name}: {data_comp.data_consistency_score:.1%} consistency"
                    )
                
                if len(data_comp.mismatched_records) > 100:
                    analysis.potential_data_quality_problems.append(
                        f"High number of mismatched records in {file_name}: {len(data_comp.mismatched_records)}"
                    )
        
        return analysis
    
    def _fallback_analysis(self, component_data: Dict[str, Any]) -> IntelligentAnalysis:
        """Fallback analysis when LLM is not available"""
        
        analysis = IntelligentAnalysis(
            component_name=component_data['name'],
            component_type=component_data['component_type']
        )
        
        # Use rule-based analysis
        analysis_data = component_data.get('analysis', {})
        
        # Determine business purpose based on patterns
        business_functions = analysis_data.get('business_functions', [])
        if 'CALCULATION' in business_functions:
            analysis.business_purpose = "Performs calculations and data processing operations"
        elif 'REPORTING' in business_functions:
            analysis.business_purpose = "Generates reports and data outputs"
        elif 'VALIDATION' in business_functions:
            analysis.business_purpose = "Validates and verifies data integrity"
        else:
            analysis.business_purpose = "Handles data processing operations"
        
        # Basic complexity assessment
        complexity_score = analysis_data.get('complexity_score', 0)
        if complexity_score > 100:
            analysis.complexity_assessment = "High complexity component requiring careful maintenance"
        elif complexity_score > 50:
            analysis.complexity_assessment = "Medium complexity with manageable maintenance needs"
        else:
            analysis.complexity_assessment = "Low complexity, straightforward maintenance"
        
        # Basic recommendations
        analysis.modernization_recommendations = [
            "Consider migrating to modern programming language",
            "Implement automated testing",
            "Review for optimization opportunities"
        ]
        
        analysis.analysis_confidence = 0.6  # Lower confidence for rule-based
        
        return analysis

class IntegratedMainframeAnalyzer:
    """Integrated analyzer combining LLM intelligence with file/DB2 comparison"""
    
    def __init__(self, db2_config: Dict[str, str] = None):
        self.llm_engine = LLMAnalysisEngine()
        self.file_db2_comparator = None
        
        if db2_config:
            from file_db2_comparison import MainframeFileDB2Comparator
            self.file_db2_comparator = MainframeFileDB2Comparator(db2_config)
    
    def perform_comprehensive_analysis(self, components: List[Dict[str, Any]], 
                                     csv_files: List[str] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis with LLM intelligence and file/DB2 comparison"""
        
        results = {
            'component_analyses': {},
            'file_db2_comparisons': {},
            'cross_component_insights': {},
            'system_recommendations': [],
            'migration_roadmap': [],
            'risk_assessment': {}
        }
        
        # Step 1: Perform file vs DB2 comparisons if CSV files provided
        if csv_files and self.file_db2_comparator:
            print("🔍 Performing file vs DB2 comparisons...")
            
            # Auto-map files to tables based on component analysis
            self._setup_file_table_mappings(components)
            
            results['file_db2_comparisons'] = self.file_db2_comparator.auto_compare_files(csv_files)
        
        # Step 2: Intelligent analysis of each component
        print("🧠 Performing LLM-powered component analysis...")
        
        for component in components:
            component_name = component['name']
            
            # Get related components for context
            related_components = self._find_related_components(component, components)
            
            # Get relevant file/DB2 comparison for this component
            relevant_comparisons = self._find_relevant_comparisons(
                component, results['file_db2_comparisons']
            )
            
            # Perform intelligent analysis
            intelligent_analysis = self.llm_engine.analyze_component_intelligence(
                component, related_components, relevant_comparisons
            )
            
            results['component_analyses'][component_name] = intelligent_analysis
        
        # Step 3: Cross-component system analysis
        results['cross_component_insights'] = self._analyze_system_level_patterns(
            results['component_analyses'], results['file_db2_comparisons']
        )
        
        # Step 4: Generate system-level recommendations
        results['system_recommendations'] = self._generate_system_recommendations(results)
        
        # Step 5: Create migration roadmap
        results['migration_roadmap'] = self._create_migration_roadmap(results)
        
        # Step 6: Overall risk assessment
        results['risk_assessment'] = self._assess_system_risk(results)
        
        return results
    
    def _setup_file_table_mappings(self, components: List[Dict[str, Any]]):
        """Setup automatic file-to-table mappings based on component analysis"""
        
        for component in components:
            analysis_data = component.get('analysis', {})
            
            # Map files created/updated by programs to potential DB2 tables
            created_files = analysis_data.get('creates_files', [])
            updated_files = analysis_data.get('updates_files', [])
            
            for file_name in created_files + updated_files:
                # Convert file name to potential table name
                table_name = file_name.replace('.DAT', '').replace('.TXT', '').replace('.CSV', '')
                table_name = table_name.replace('-', '_')  # DB2 naming convention
                
                self.file_db2_comparator.add_mapping_rule(file_name, table_name)
    
    def _find_related_components(self, component: Dict[str, Any], 
                               all_components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find components related to the current component"""
        
        related = []
        component_analysis = component.get('analysis', {})
        component_files = set(
            component_analysis.get('creates_files', []) +
            component_analysis.get('reads_files', []) +
            component_analysis.get('updates_files', [])
        )
        
        for other_component in all_components:
            if other_component['name'] == component['name']:
                continue
            
            other_analysis = other_component.get('analysis', {})
            other_files = set(
                other_analysis.get('creates_files', []) +
                other_analysis.get('reads_files', []) +
                other_analysis.get('updates_files', [])
            )
            
            # Check for shared files
            if component_files & other_files:
                related.append(other_component)
            
            # Check for program calls
            called_programs = component_analysis.get('calls_programs', set())
            if other_component['name'] in called_programs:
                related.append(other_component)
        
        return related
    
    def _find_relevant_comparisons(self, component: Dict[str, Any], 
                                 file_db2_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Find file/DB2 comparisons relevant to the component"""
        
        relevant = {}
        component_analysis = component.get('analysis', {})
        component_files = set(
            component_analysis.get('creates_files', []) +
            component_analysis.get('reads_files', []) +
            component_analysis.get('updates_files', [])
        )
        
        for file_name, comparison in file_db2_comparisons.items():
            # Check if any component files match the comparison file
            for comp_file in component_files:
                if comp_file in file_name or file_name in comp_file:
                    relevant[file_name] = comparison
                    break
        
        return relevant
    
    def _analyze_system_level_patterns(self, component_analyses: Dict[str, IntelligentAnalysis],
                                     file_db2_comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system-level patterns and insights"""
        
        insights = {
            'common_data_quality_issues': [],
            'architectural_patterns': [],
            'modernization_priorities': [],
            'data_flow_complexity': {}
        }
        
        # Analyze common issues across components
        all_issues = []
        for analysis in component_analyses.values():
            all_issues.extend(analysis.data_consistency_issues)
            all_issues.extend(analysis.potential_data_quality_problems)
        
        # Find most common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        insights['common_data_quality_issues'] = [
            f"{issue} (affects {count} components)"
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Analyze architectural patterns
        high_risk_components = [
            name for name, analysis in component_analyses.items()
            if analysis.change_risk_level == 'HIGH'
        ]
        
        if len(high_risk_components) > len(component_analyses) * 0.3:
            insights['architectural_patterns'].append(
                "High-risk architecture: Many components have high change risk"
            )
        
        # Prioritize modernization based on risk and impact
        modernization_scores = {}
        for name, analysis in component_analyses.items():
            risk_score = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}.get(analysis.change_risk_level, 2)
            impact_score = len(analysis.downstream_impacts)
            modernization_scores[name] = risk_score + impact_score
        
        insights['modernization_priorities'] = [
            name for name, score in sorted(modernization_scores.items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
        ]
        
        return insights
    
    def _generate_system_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate system-level recommendations based on comprehensive analysis"""
        
        recommendations = []
        component_analyses = results['component_analyses']
        file_db2_comparisons = results['file_db2_comparisons']
        cross_insights = results['cross_component_insights']
        
        # Data quality recommendations
        if any('consistency' in issue.lower() for issue in cross_insights.get('common_data_quality_issues', [])):
            recommendations.append(
                "🔧 CRITICAL: Implement comprehensive data validation framework across all components"
            )
        
        # Schema standardization
        schema_issues = sum(1 for comp in file_db2_comparisons.values() 
                          if comp.get('schema_comparison', {}).get('compatibility_score', 1) < 0.8)
        
        if schema_issues > 0:
            recommendations.append(
                f"📋 HIGH: Standardize schemas - {schema_issues} files have compatibility issues"
            )
        
        # High-risk component handling
        high_risk_components = [
            name for name, analysis in component_analyses.items()
            if analysis.change_risk_level == 'HIGH'
        ]
        
        if high_risk_components:
            recommendations.append(
                f"⚠️ HIGH: Address high-risk components first: {', '.join(high_risk_components[:3])}"
            )
        
        # Modernization strategy
        if len(cross_insights.get('modernization_priorities', [])) > 0:
            recommendations.append(
                "🚀 MEDIUM: Start modernization with highest-priority components identified in analysis"
            )
        
        return recommendations
    
    def _create_migration_roadmap(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a phased migration roadmap"""
        
        roadmap = []
        component_analyses = results['component_analyses']
        
        # Phase 1: Low-risk, high-impact components
        phase1_components = [
            name for name, analysis in component_analyses.items()
            if analysis.change_risk_level == 'LOW' and len(analysis.downstream_impacts) > 2
        ]
        
        if phase1_components:
            roadmap.append({
                'phase': 1,
                'title': 'Quick Wins - Low Risk, High Impact',
                'components': phase1_components,
                'duration': '2-3 months',
                'description': 'Start with low-risk components that provide high business value'
            })
        
        # Phase 2: Schema standardization
        roadmap.append({
            'phase': 2,
            'title': 'Schema Standardization',
            'components': ['Schema Migration', 'Data Quality Framework'],
            'duration': '3-4 months',
            'description': 'Standardize schemas and implement data quality controls'
        })
        
        # Phase 3: High-risk components
        high_risk_components = [
            name for name, analysis in component_analyses.items()
            if analysis.change_risk_level == 'HIGH'
        ]
        
        if high_risk_components:
            roadmap.append({
                'phase': 3,
                'title': 'High-Risk Component Migration',
                'components': high_risk_components,
                'duration': '6-8 months',
                'description': 'Carefully migrate high-risk, business-critical components'
            })
        
        return roadmap
    
    def _assess_system_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system risk"""
        
        component_analyses = results['component_analyses']
        file_db2_comparisons = results['file_db2_comparisons']
        
        # Calculate risk metrics
        total_components = len(component_analyses)
        high_risk_count = sum(1 for analysis in component_analyses.values() 
                            if analysis.change_risk_level == 'HIGH')
        
        low_schema_compatibility = sum(1 for comp in file_db2_comparisons.values()
                                     if comp.get('schema_comparison', {}).get('compatibility_score', 1) < 0.7)
        
        low_data_consistency = sum(1 for comp in file_db2_comparisons.values()
                                 if comp.get('data_comparison', {}).get('data_consistency_score', 1) < 0.8)
        
        # Overall risk assessment
        risk_score = 0
        risk_factors = []
        
        if high_risk_count > total_components * 0.3:
            risk_score += 3
            risk_factors.append(f"High proportion of risky components ({high_risk_count}/{total_components})")
        
        if low_schema_compatibility > 0:
            risk_score += 2
            risk_factors.append(f"Schema compatibility issues in {low_schema_compatibility} files")
        
        if low_data_consistency > 0:
            risk_score += 2
            risk_factors.append(f"Data consistency issues in {low_data_consistency} files")
        
        # Determine overall risk level
        if risk_score >= 6:
            overall_risk = "HIGH"
        elif risk_score >= 3:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        return {
            'overall_risk_level': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'high_risk_components': high_risk_count,
            'total_components': total_components,
            'schema_issues': low_schema_compatibility,
            'data_issues': low_data_consistency,
            'confidence_level': sum(analysis.analysis_confidence for analysis in component_analyses.values()) / total_components if total_components > 0 else 0
        }

# Integration with Streamlit UI
def create_intelligent_analysis_ui():
    """Enhanced Streamlit UI with LLM intelligence and integrated file/DB2 comparison"""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.title("🧠 Opulence Intelligent Mainframe Analysis System")
    st.markdown("### LLM-Powered Analysis with File vs DB2 Integration")
    
    # Initialize analyzer
    if 'integrated_analyzer' not in st.session_state:
        # DB2 Configuration
        st.sidebar.header("🗄️ DB2 Configuration")
        db2_host = st.sidebar.text_input("DB2 Host")
        db2_database = st.sidebar.text_input("Database")
        db2_username = st.sidebar.text_input("Username")
        db2_password = st.sidebar.text_input("Password", type="password")
        
        db2_config = None
        if all([db2_host, db2_database, db2_username, db2_password]):
            db2_config = {
                'hostname': db2_host,
                'port': '50000',
                'database': db2_database,
                'username': db2_username,
                'password': db2_password
            }
        
        st.session_state.integrated_analyzer = IntegratedMainframeAnalyzer(db2_config)
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Mainframe Components")
        component_files = st.file_uploader(
            "Upload mainframe source files",
            accept_multiple_files=True,
            type=['cbl', 'cob', 'cobol', 'jcl', 'job', 'cpy', 'copy', 'sql']
        )
    
    with col2:
        st.subheader("📊 Data Files (CSV)")
        csv_files = st.file_uploader(
            "Upload CSV data files for comparison",
            accept_multiple_files=True,
            type=['csv']
        )
    
    # Analysis options
    st.subheader("⚙️ Analysis Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_llm = st.checkbox("Use LLM Intelligence", value=True, 
                             help="Use CodeLlama for intelligent analysis")
        deep_analysis = st.checkbox("Deep Cross-Component Analysis", value=True)
    
    with col2:
        include_file_db2 = st.checkbox("Include File vs DB2 Comparison", value=True)
        generate_roadmap = st.checkbox("Generate Migration Roadmap", value=True)
    
    with col3:
        analysis_depth = st.selectbox("Analysis Depth", 
                                    ["Standard", "Comprehensive", "Expert"])
        sample_size = st.slider("DB2 Sample Size", 1000, 25000, 10000)
    
    # Run comprehensive analysis
    if st.button("🚀 Start Intelligent Analysis", type="primary"):
        if not component_files:
            st.error("Please upload at least one mainframe component file")
            return
        
        with st.spinner("🧠 Performing intelligent analysis..."):
            # Process component files
            components = []
            temp_component_files = []
            
            for uploaded_file in component_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_component_files.append(temp_path)
                
                # Read content for analysis
                content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
                components.append({
                    'name': uploaded_file.name,
                    'path': temp_path,
                    'content': content,
                    'component_type': 'PROGRAM',  # Will be auto-detected
                    'analysis': {}  # Will be populated by batch processor
                })
            
            # Process CSV files
            csv_file_paths = []
            if csv_files and include_file_db2:
                for csv_file in csv_files:
                    temp_path = f"temp_{csv_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(csv_file.getbuffer())
                    csv_file_paths.append(temp_path)
            
            # Perform comprehensive analysis
            try:
                results = st.session_state.integrated_analyzer.perform_comprehensive_analysis(
                    components, csv_file_paths
                )
                
                # Store results in session
                st.session_state.analysis_results = results
                
                # Display results
                display_intelligent_analysis_results(results)
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)
            
            finally:
                # Clean up temp files
                for temp_path in temp_component_files + csv_file_paths:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

def display_intelligent_analysis_results(results: Dict[str, Any]):
    """Display comprehensive analysis results with intelligence insights"""
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    
    st.success("✅ Intelligent analysis completed!")
    
    # Executive Summary Dashboard
    st.header("📈 Executive Summary")
    
    component_analyses = results['component_analyses']
    file_db2_comparisons = results['file_db2_comparisons']
    risk_assessment = results['risk_assessment']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Components Analyzed", len(component_analyses))
        st.metric("Overall Risk Level", risk_assessment['overall_risk_level'],
                 delta=f"Confidence: {risk_assessment['confidence_level']:.1%}")
    
    with col2:
        high_risk_count = risk_assessment['high_risk_components']
        st.metric("High-Risk Components", high_risk_count)
        
        if file_db2_comparisons:
            avg_schema_compatibility = sum(
                comp.get('schema_comparison', {}).get('compatibility_score', 0)
                for comp in file_db2_comparisons.values()
                if 'schema_comparison' in comp
            ) / len(file_db2_comparisons)
            st.metric("Avg Schema Compatibility", f"{avg_schema_compatibility:.1%}")
    
    with col3:
        if file_db2_comparisons:
            data_comparisons = [comp.get('data_comparison') for comp in file_db2_comparisons.values()]
            data_comparisons = [comp for comp in data_comparisons if comp]
            
            if data_comparisons:
                avg_data_consistency = sum(comp.get('data_consistency_score', 0) for comp in data_comparisons) / len(data_comparisons)
                st.metric("Avg Data Consistency", f"{avg_data_consistency:.1%}")
    
    with col4:
        system_recommendations = len(results['system_recommendations'])
        st.metric("System Recommendations", system_recommendations)
        
        migration_phases = len(results['migration_roadmap'])
        st.metric("Migration Phases", migration_phases)
    
    # Risk Assessment Visualization
    st.subheader("🎯 Risk Assessment Overview")
    
    risk_data = []
    for name, analysis in component_analyses.items():
        risk_score = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}.get(analysis.change_risk_level, 2)
        risk_data.append({
            'Component': name,
            'Risk Level': analysis.change_risk_level,
            'Risk Score': risk_score,
            'Confidence': analysis.analysis_confidence,
            'Downstream Impacts': len(analysis.downstream_impacts)
        })
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data)
        
        fig = px.scatter(risk_df, 
                        x='Risk Score', 
                        y='Downstream Impacts',
                        size='Confidence',
                        color='Risk Level',
                        hover_name='Component',
                        title='Component Risk vs Impact Analysis',
                        color_discrete_map={'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 LLM Insights", "📊 File vs DB2", "🔗 Cross-Component", "🗺️ Migration Roadmap", "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("🧠 LLM-Generated Insights")
        
        for component_name, analysis in component_analyses.items():
            with st.expander(f"📋 {component_name} - {analysis.change_risk_level} Risk"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Business Purpose:**")
                    st.write(analysis.business_purpose)
                    
                    st.write("**Data Flow:**")
                    st.write(analysis.data_flow_description)
                    
                    st.write("**Complexity Assessment:**")
                    st.write(analysis.complexity_assessment)
                
                with col2:
                    st.write("**Modernization Recommendations:**")
                    for rec in analysis.modernization_recommendations:
                        st.write(f"• {rec}")
                    
                    st.write("**Risk Factors:**")
                    for risk in analysis.risk_factors:
                        st.write(f"⚠️ {risk}")
                    
                    st.write("**Mitigation Strategies:**")
                    for strategy in analysis.mitigation_strategies:
                        st.write(f"✅ {strategy}")
                
                # LLM reasoning
                if analysis.llm_reasoning:
                    st.write("**LLM Analysis Reasoning:**")
                    st.info(analysis.llm_reasoning)
                
                # Confidence indicator
                confidence_color = "green" if analysis.analysis_confidence > 0.8 else "orange" if analysis.analysis_confidence > 0.6 else "red"
                st.markdown(f"**Analysis Confidence:** <span style='color: {confidence_color}'>{analysis.analysis_confidence:.1%}</span>", 
                           unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📊 File vs DB2 Comparison Results")
        
        if file_db2_comparisons:
            # Summary metrics
            comparison_summary = []
            for file_name, comparison in file_db2_comparisons.items():
                if 'error' in comparison:
                    comparison_summary.append({
                        'File': file_name,
                        'Status': 'Error',
                        'Schema Compatibility': 'N/A',
                        'Data Consistency': 'N/A',
                        'Issues': comparison['error']
                    })
                else:
                    schema_comp = comparison.get('schema_comparison')
                    data_comp = comparison.get('data_comparison')
                    
                    comparison_summary.append({
                        'File': file_name,
                        'Status': 'Success',
                        'Schema Compatibility': f"{schema_comp.compatibility_score:.1%}" if schema_comp else 'N/A',
                        'Data Consistency': f"{data_comp.data_consistency_score:.1%}" if data_comp else 'N/A',
                        'Schema Issues': schema_comp.total_issues if schema_comp else 0,
                        'Data Issues': len(data_comp.mismatched_records) if data_comp else 0
                    })
            
            summary_df = pd.DataFrame(comparison_summary)
            st.dataframe(summary_df, use_container_width=True)
            
            # Detailed comparison for each file
            for file_name, comparison in file_db2_comparisons.items():
                if 'error' not in comparison:
                    with st.expander(f"📁 {file_name} Detailed Analysis"):
                        schema_comp = comparison.get('schema_comparison')
                        data_comp = comparison.get('data_comparison')
                        recommendations = comparison.get('recommendations', [])
                        
                        if schema_comp:
                            st.write("**Schema Analysis:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Common Columns", len(schema_comp.common_columns))
                            with col2:
                                st.metric("File Only", len(schema_comp.file_only_columns))
                            with col3:
                                st.metric("DB2 Only", len(schema_comp.db2_only_columns))
                            
                            if schema_comp.type_mismatches:
                                st.write("**Type Mismatches:**")
                                mismatch_df = pd.DataFrame(schema_comp.type_mismatches)
                                st.dataframe(mismatch_df)
                        
                        if data_comp:
                            st.write("**Data Analysis:**")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("File Records", data_comp.total_file_records)
                            with col2:
                                st.metric("DB2 Records", data_comp.total_db2_records)
                            with col3:
                                st.metric("Matching Records", data_comp.matching_records)
                            
                            if data_comp.field_differences:
                                st.write("**Field Differences:**")
                                for field, diffs in data_comp.field_differences.items():
                                    st.write(f"• **{field}**: {len(diffs)} differences")
                        
                        if recommendations:
                            st.write("**Recommendations:**")
                            for rec in recommendations:
                                st.write(f"💡 {rec}")
        else:
            st.info("No file vs DB2 comparisons performed. Upload CSV files to enable this analysis.")
    
    with tab3:
        st.subheader("🔗 Cross-Component Analysis")
        
        cross_insights = results['cross_component_insights']
        
        st.write("**Common Data Quality Issues:**")
        for issue in cross_insights.get('common_data_quality_issues', []):
            st.write(f"⚠️ {issue}")
        
        st.write("**Architectural Patterns:**")
        for pattern in cross_insights.get('architectural_patterns', []):
            st.write(f"🏗️ {pattern}")
        
        st.write("**Modernization Priorities:**")
        priorities = cross_insights.get('modernization_priorities', [])
        for i, component in enumerate(priorities, 1):
            st.write(f"{i}. **{component}**")
        
        # Component dependency graph
        st.subheader("📊 Component Relationships")
        
        # Create network graph data
        nodes = []
        edges = []
        
        for name, analysis in component_analyses.items():
            # Add node
            risk_color = {'LOW': 'green', 'MEDIUM': 'orange', 'HIGH': 'red'}.get(analysis.change_risk_level, 'gray')
            nodes.append({
                'id': name,
                'label': name,
                'color': risk_color,
                'size': len(analysis.downstream_impacts) * 10 + 20
            })
            
            # Add edges for dependencies
            for dep in analysis.upstream_dependencies:
                edges.append({'from': dep, 'to': name})
        
        if nodes:
            st.write("**Component Dependency Visualization:**")
            st.write("🟢 Low Risk | 🟠 Medium Risk | 🔴 High Risk")
            
            # Simple network representation
            dependency_data = []
            for name, analysis in component_analyses.items():
                dependency_data.append({
                    'Component': name,
                    'Risk Level': analysis.change_risk_level,
                    'Dependencies': len(analysis.upstream_dependencies),
                    'Impacts': len(analysis.downstream_impacts),
                    'Shared Data': len(analysis.shared_data_patterns)
                })
            
            if dependency_data:
                dep_df = pd.DataFrame(dependency_data)
                st.dataframe(dep_df, use_container_width=True)
    
    with tab4:
        st.subheader("🗺️ Migration Roadmap")
        
        roadmap = results['migration_roadmap']
        
        if roadmap:
            for phase in roadmap:
                with st.expander(f"📅 Phase {phase['phase']}: {phase['title']} ({phase['duration']})"):
                    st.write(f"**Description:** {phase['description']}")
                    
                    st.write("**Components/Activities:**")
                    for component in phase['components']:
                        st.write(f"• {component}")
                    
                    # Progress indicator (placeholder)
                    st.progress(0.0)  # Would be updated based on actual progress
            
            # Timeline visualization
            st.subheader("📈 Migration Timeline")
            
            timeline_data = []
            start_month = 0
            
            for phase in roadmap:
                duration_months = int(phase['duration'].split('-')[0])  # Extract first number
                timeline_data.append({
                    'Phase': f"Phase {phase['phase']}",
                    'Start': start_month,
                    'Duration': duration_months,
                    'End': start_month + duration_months,
                    'Title': phase['title']
                })
                start_month += duration_months
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                
                fig = px.timeline(timeline_df, 
                                x_start='Start', 
                                x_end='End',
                                y='Phase',
                                title='Migration Timeline',
                                hover_data=['Title'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No migration roadmap generated. Enable roadmap generation in analysis options.")
    
    with tab5:
        st.subheader("💡 System Recommendations")
        
        system_recommendations = results['system_recommendations']
        
        if system_recommendations:
            st.write("**Priority Recommendations:**")
            for i, rec in enumerate(system_recommendations, 1):
                priority = "🔴 HIGH" if "CRITICAL" in rec else "🟠 MEDIUM" if "HIGH" in rec else "🟡 LOW"
                st.write(f"{i}. {priority}: {rec}")
        
        # Action plan generator
        st.subheader("📋 Generated Action Plan")
        
        if st.button("🎯 Generate Detailed Action Plan"):
            with st.spinner("Generating action plan..."):
                # This could use LLM to generate a detailed action plan
                action_plan = generate_action_plan(results)
                
                st.write("**Immediate Actions (Next 30 days):**")
                for action in action_plan.get('immediate', []):
                    st.write(f"☑️ {action}")
                
                st.write("**Short-term Actions (Next 90 days):**")
                for action in action_plan.get('short_term', []):
                    st.write(f"📅 {action}")
                
                st.write("**Long-term Strategy (Next 12 months):**")
                for action in action_plan.get('long_term', []):
                    st.write(f"🎯 {action}")

def generate_action_plan(results: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate detailed action plan based on analysis results"""
    
    action_plan = {
        'immediate': [],
        'short_term': [],
        'long_term': []
    }
    
    # Immediate actions based on high-risk issues
    risk_assessment = results['risk_assessment']
    if risk_assessment['overall_risk_level'] == 'HIGH':
        action_plan['immediate'].append("Establish emergency response team for high-risk components")
        action_plan['immediate'].append("Create backup and rollback procedures")
    
    if risk_assessment['schema_issues'] > 0:
        action_plan['immediate'].append("Audit and document all schema inconsistencies")
    
    if risk_assessment['data_issues'] > 0:
        action_plan['immediate'].append("Implement data validation checks")
    
    # Short-term actions
    component_analyses = results['component_analyses']
    high_risk_components = [name for name, analysis in component_analyses.items() 
                          if analysis.change_risk_level == 'HIGH']
    
    if high_risk_components:
        action_plan['short_term'].append(f"Detailed analysis of high-risk components: {', '.join(high_risk_components[:3])}")
    
    action_plan['short_term'].append("Implement automated testing for critical components")
    action_plan['short_term'].append("Create comprehensive documentation for all components")
    
    # Long-term actions
    if results['migration_roadmap']:
        action_plan['long_term'].append("Execute phased migration according to generated roadmap")
    
    action_plan['long_term'].append("Implement modern CI/CD pipeline")
    action_plan['long_term'].append("Train team on new technologies and practices")
    
    return action_plan

# Example usage and testing
def example_intelligent_analysis():
    """Example of running intelligent analysis"""
    
    # Sample components for testing
    sample_components = [
        {
            'name': 'PAYROLL.CBL',
            'content': '''
            IDENTIFICATION DIVISION.
            PROGRAM-ID. PAYROLL.
            
            DATA DIVISION.
            FILE SECTION.
            FD EMPLOYEE-FILE.
            01 EMPLOYEE-RECORD.
               05 EMP-ID PIC 9(6).
               05 EMP-SALARY PIC 9(7)V99.
            
            PROCEDURE DIVISION.
            OPEN INPUT EMPLOYEE-FILE
            READ EMPLOYEE-FILE
            COMPUTE PAY-AMOUNT = EMP-SALARY * 0.8
            STOP RUN.
            ''',
            'component_type': 'PROGRAM',
            'analysis': {
                'file_operations': [{'operation': 'READ', 'file_name': 'EMPLOYEE-FILE'}],
                'fields_defined': [{'name': 'EMP-ID'}, {'name': 'EMP-SALARY'}],
                'complexity_score': 45.5,
                'business_functions': ['CALCULATION']
            }
        }
    ]
    
    # Sample CSV files
    sample_csv_files = ['employee_data.csv', 'payroll_data.csv']
    
    # Initialize analyzer
    analyzer = IntegratedMainframeAnalyzer()
    
    # Run analysis
    results = analyzer.perform_comprehensive_analysis(sample_components, sample_csv_files)
    
    print("=== Intelligent Analysis Results ===")
    print(f"Components analyzed: {len(results['component_analyses'])}")
    print(f"System recommendations: {len(results['system_recommendations'])}")
    print(f"Migration phases: {len(results['migration_roadmap'])}")
    print(f"Overall risk: {results['risk_assessment']['overall_risk_level']}")

if __name__ == "__main__":
    print("🧠 LLM-Powered Intelligent Mainframe Analysis System")
    print("=" * 60)
    
    print("LLM Intelligence Features:")
    print("✅ Business purpose identification")
    print("✅ Data flow analysis")
    print("✅ Complexity assessment")
    print("✅ Modernization recommendations")
    print("✅ Risk assessment with mitigation strategies")
    print("✅ Cross-component relationship analysis")
    print("✅ Integration with file vs DB2 comparison")
    print("✅ Intelligent migration roadmap generation")
    print("✅ System-level insights and patterns")
    print("✅ Confidence scoring for all analyses")
    
    print("\nFile vs DB2 Integration:")
    print("✅ Schema compatibility analysis")
    print("✅ Data consistency scoring")
    print("✅ Intelligent mapping of files to tables")
    print("✅ Root cause analysis of data issues")
    print("✅ Migration strategy recommendations")
    
    print("\nTo use:")
    print("1. Configure DB2 connection")
    print("2. Upload mainframe components and CSV files")
    print("3. Enable LLM intelligence")
    print("4. Run comprehensive analysis")
    print("5. Review LLM-generated insights")
    print("6. Follow migration roadmap")
    
    # Run example
    # example_intelligent_analysis()