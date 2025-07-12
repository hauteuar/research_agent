# agents/base_agent_api.py
"""
API-Compatible Base Agent for Opulence System - FIXED VERSION
Provides common functionality for all agents using API calls instead of direct GPU management
KEEPS ALL CLASS NAMES AND FUNCTION SIGNATURES FOR COMPATIBILITY
"""

import weakref
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, Optional
import time
import json

class BaseOpulenceAgent:
    """Base class for all API-based Opulence agents - EXACT SAME CLASS NAME"""
    
    def __init__(self, coordinator, agent_type: str, db_path: str = "opulence_data.db", gpu_id: int = 0):
        self.coordinator = coordinator
        self.agent_type = agent_type
        self.db_path = db_path
        self.gpu_id = gpu_id  # Keep for compatibility but not used for GPU selection
        self.logger = logging.getLogger(f"{__name__}.{agent_type}")
        
        # API-specific tracking
        self._api_calls_made = 0
        self._total_api_time = 0.0
        self._last_api_call_time = None
        self._initialization_time = time.time()
        
        # Agent state
        self._is_active = True
        
        # API parameters
        self.api_params = {
            "max_tokens": 1000,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 50,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None,
            "stream": False,
            "seed": None
        }
        
        # Register for automatic cleanup
        if coordinator:
            weakref.finalize(self, self._cleanup_callback, agent_type)
        
        self.logger.info(f"Initialized {agent_type} agent (API-based)")
    
    @staticmethod
    def _cleanup_callback(agent_type: str):
        """Static cleanup callback for weakref"""
        try:
            logging.getLogger(__name__).info(f"🧹 Auto-cleaned {agent_type} agent")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Auto-cleanup failed for {agent_type}: {e}")
    
    @asynccontextmanager
    async def get_engine_context(self):
        """API-compatible context manager - returns API client interface"""
        try:
            # Create API context wrapper
            api_context = APIEngineContext(self.coordinator, self.gpu_id)
            yield api_context
        except Exception as e:
            self.logger.error(f"API context error in {self.agent_type}: {e}")
            raise
        finally:
            # No cleanup needed for API calls
            pass
    
    async def call_api(self, prompt: str, params: Dict[str, Any] = None) -> str:
        """Make API call through coordinator with tracking"""
        start_time = time.time()
        
        try:
            # Merge agent-specific params with call-specific params
            final_params = self.api_params.copy()
            if params:
                final_params.update(params)
            
            # Clean parameters
            cleaned_params = {}
            for key, value in final_params.items():
                if value is not None:
                    if key == "max_tokens":
                        cleaned_params[key] = max(1, min(value, 4096))
                    elif key == "temperature":
                        cleaned_params[key] = max(0.0, min(value, 2.0))
                    elif key == "top_p":
                        cleaned_params[key] = max(0.0, min(value, 1.0))
                    elif key == "top_k":
                        cleaned_params[key] = max(1, min(value, 100))
                    elif key in ["frequency_penalty", "presence_penalty"]:
                        cleaned_params[key] = max(-2.0, min(value, 2.0))
                    else:
                        cleaned_params[key] = value
            
            # Make API call through coordinator - GPU ID is ignored in coordinator
            result = await self.coordinator.call_model_api(
                prompt=prompt,
                params=cleaned_params,
                preferred_gpu_id=self.gpu_id  # This will be ignored by coordinator
            )
            
            # Extract response text from various response formats
            if isinstance(result, dict):
                response_text = (
                    result.get('text') or 
                    result.get('response') or 
                    result.get('content') or
                    result.get('generated_text') or
                    str(result.get('choices', [{}])[0].get('text', '')) or
                    ''
                )
            else:
                response_text = str(result)
            
            # Track API call statistics
            api_time = time.time() - start_time
            self._api_calls_made += 1
            self._total_api_time += api_time
            self._last_api_call_time = time.time()
            
            self.logger.debug(f"API call completed in {api_time:.2f}s for {self.agent_type}")
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"API call failed for {self.agent_type}: {str(e)}")
            # More specific error handling
            if "422" in str(e):
                raise RuntimeError(f"API validation error - check parameters: {str(e)}")
            elif "503" in str(e):
                raise RuntimeError(f"Model server unavailable: {str(e)}")
            elif "timeout" in str(e).lower():
                raise RuntimeError(f"API call timeout: {str(e)}")
            else:
                raise RuntimeError(f"API call failed: {str(e)}")
    
    async def get_engine(self):
        """DEPRECATED: For backwards compatibility - use get_engine_context() instead"""
        self.logger.warning(f"⚠️ {self.agent_type} using deprecated get_engine() - use call_api() instead")
        
        # Return a mock engine that delegates to API calls
        return MockEngine(self)
    
    def _add_processing_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add processing information to results"""
        if isinstance(result, dict):
            result.update({
                'gpu_used': self.gpu_id,  # Keep for compatibility
                'agent_type': self.agent_type,
                'api_based': True,
                'api_calls_made': self._api_calls_made,
                'total_api_time': self._total_api_time,
                'average_api_time': self._total_api_time / max(self._api_calls_made, 1),
                'coordinator_type': getattr(self.coordinator, 'stats', {}).get('coordinator_type', 'api_based')
            })
        return result
    
    def cleanup(self):
        """Manual cleanup method"""
        self.logger.info(f"🧹 Cleaning up {self.agent_type} agent...")
        
        # Mark as inactive
        self._is_active = False
        
        # Log final statistics
        if self._api_calls_made > 0:
            avg_time = self._total_api_time / self._api_calls_made
            uptime = time.time() - self._initialization_time
            self.logger.info(f"✅ {self.agent_type} final stats: {self._api_calls_made} API calls, "
                           f"avg {avg_time:.2f}s, uptime {uptime:.1f}s")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        uptime = time.time() - self._initialization_time
        avg_api_time = self._total_api_time / max(self._api_calls_made, 1)
        
        return {
            "agent_type": self.agent_type,
            "gpu_id": self.gpu_id,
            "api_based": True,
            "is_active": self._is_active,
            "api_calls_made": self._api_calls_made,
            "total_api_time": self._total_api_time,
            "average_api_time": avg_api_time,
            "last_api_call_time": self._last_api_call_time,
            "uptime_seconds": uptime,
            "calls_per_minute": (self._api_calls_made / max(uptime / 60, 1)) if uptime > 0 else 0
        }
    
    def update_api_params(self, **kwargs):
        """Update API parameters for this agent"""
        self.api_params.update(kwargs)
        self.logger.info(f"Updated API params for {self.agent_type}: {kwargs}")
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"type={self.agent_type}, "
                f"gpu={self.gpu_id}, "
                f"api_calls={self._api_calls_made}, "
                f"active={self._is_active})")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


class APIEngineContext:
    """API-compatible engine context that mimics the original engine interface"""
    
    def __init__(self, coordinator, preferred_gpu_id: int = None):
        self.coordinator = coordinator
        self.preferred_gpu_id = preferred_gpu_id  # Keep for compatibility but will be ignored
        self.logger = logging.getLogger(f"{__name__}.APIEngineContext")
    
    async def generate(self, prompt: str, sampling_params, request_id: str = None):
        """Generate text via API (compatible with original engine interface)"""
        # Convert sampling_params to dict
        params = {}
        
        if hasattr(sampling_params, '__dict__'):
            # Object with attributes
            for attr in ['max_tokens', 'temperature', 'top_p', 'top_k', 
                        'frequency_penalty', 'presence_penalty', 'stop', 'seed']:
                if hasattr(sampling_params, attr):
                    value = getattr(sampling_params, attr)
                    if value is not None:
                        params[attr] = value
        elif isinstance(sampling_params, dict):
            # Dictionary
            params = sampling_params.copy()
        else:
            # Fallback defaults
            params = {
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9
            }
        
        # Validate parameter ranges
        validated_params = {}
        for key, value in params.items():
            if value is not None:
                if key == "max_tokens":
                    validated_params[key] = max(1, min(value, 4096))
                elif key == "temperature":
                    validated_params[key] = max(0.0, min(value, 2.0))
                elif key == "top_p":
                    validated_params[key] = max(0.0, min(value, 1.0))
                elif key == "top_k":
                    validated_params[key] = max(1, min(value, 100))
                elif key in ["frequency_penalty", "presence_penalty"]:
                    validated_params[key] = max(-2.0, min(value, 2.0))
                else:
                    validated_params[key] = value
        
        # Call API - preferred_gpu_id will be ignored by coordinator
        result = await self.coordinator.call_model_api(
            prompt=prompt, 
            params=validated_params, 
            preferred_gpu_id=self.preferred_gpu_id
        )
        
        # Convert response format for compatibility
        class MockOutput:
            def __init__(self, text: str, finish_reason: str):
                self.text = text
                self.finish_reason = finish_reason
                self.token_ids = []  # Not available from API
        
        class MockRequestOutput:
            def __init__(self, result: Dict[str, Any]):
                # Extract text from various possible response formats
                if isinstance(result, dict):
                    text = (
                        result.get('text') or 
                        result.get('response') or 
                        result.get('content') or
                        result.get('generated_text') or
                        ''
                    )
                    finish_reason = result.get('finish_reason', 'stop')
                else:
                    text = str(result)
                    finish_reason = 'stop'
                
                self.outputs = [MockOutput(text, finish_reason)]
                self.finished = True
                self.prompt_token_ids = []  # Not available from API
        
        # Yield result (to match async generator interface)
        yield MockRequestOutput(result)


class MockEngine:
    """Mock engine for backwards compatibility with agents expecting engine interface"""
    
    def __init__(self, agent):
        self.agent = agent
    
    async def generate(self, prompt: str, sampling_params=None, request_id: str = None):
        """Mock generate method that uses API calls"""
        # Convert parameters
        params = {}
        if sampling_params:
            if hasattr(sampling_params, '__dict__'):
                # Object with attributes
                for attr in ['max_tokens', 'temperature', 'top_p', 'top_k']:
                    if hasattr(sampling_params, attr):
                        value = getattr(sampling_params, attr)
                        if value is not None:
                            params[attr] = value
            elif isinstance(sampling_params, dict):
                params = sampling_params.copy()
        
        # Use agent's API call method
        response_text = await self.agent.call_api(prompt, params)
        
        # Mock the expected response format
        class MockOutput:
            def __init__(self, text):
                self.text = text
                self.finish_reason = "stop"
                self.token_ids = []
                
        class MockRequestOutput:
            def __init__(self, text):
                self.outputs = [MockOutput(text)]
                self.finished = True
                self.prompt_token_ids = []
        
        yield MockRequestOutput(response_text)


# ==================== Utility Functions (Keep Same Names) ====================

def create_api_agent(coordinator, agent_class, agent_type: str, **kwargs):
    """Factory function to create API-based agents"""
    return agent_class(
        coordinator=coordinator,
        llm_engine=None,  # No direct engine for API agents
        **kwargs
    )

async def test_api_agent_functionality(coordinator):
    """Test function to verify API agent functionality"""
    
    # Create a test agent
    test_agent = BaseOpulenceAgent(coordinator, "test_agent", gpu_id=2)
    
    try:
        # Test API call with various parameter formats
        response1 = await test_agent.call_api("What is 2+2?", {"max_tokens": 50, "temperature": 0.5})
        print(f"API Response 1: {response1}")
        
        # Test with edge case parameters
        response2 = await test_agent.call_api("Explain Python", {
            "max_tokens": 100,
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40
        })
        print(f"API Response 2: {response2[:100]}...")
        
        # Test context manager
        async with test_agent.get_engine_context() as engine_context:
            print("Engine context acquired successfully")
            
            # Test the mock sampling params
            class MockSamplingParams:
                def __init__(self):
                    self.max_tokens = 50
                    self.temperature = 0.3
                    self.top_p = 0.9
            
            sampling_params = MockSamplingParams()
            async for output in engine_context.generate("Hello world", sampling_params):
                print(f"Engine context response: {output.outputs[0].text}")
        
        # Get stats
        stats = test_agent.get_agent_stats()
        print(f"Agent stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        test_agent.cleanup()

# ==================== Configuration Validation ====================

async def validate_agent_server_integration(coordinator, server_endpoints):
    """Validate that agents can properly communicate with model servers"""
    
    print("🔍 Validating Agent-Server Integration...")
    
    validation_results = {
        "server_connectivity": {},
        "parameter_validation": {},
        "response_format": {},
        "error_handling": {}
    }
    
    # Test with coordinator's available servers
    available_servers = coordinator.load_balancer.get_available_servers()
    
    if not available_servers:
        print("❌ No available servers found!")
        return validation_results
    
    for server in available_servers:
        server_name = server.config.name
        print(f"\n📡 Testing server {server_name} at {server.config.endpoint}...")
        
        try:
            # Create test agent
            test_agent = BaseOpulenceAgent(coordinator, f"test_agent_{server_name}")
            
            # Test 1: Basic connectivity
            try:
                response = await test_agent.call_api("Hello", {"max_tokens": 10})
                validation_results["server_connectivity"][server_name] = {
                    "status": "✅ Connected",
                    "response_length": len(response)
                }
                print(f"  ✅ Basic connectivity: OK")
            except Exception as e:
                validation_results["server_connectivity"][server_name] = {
                    "status": "❌ Failed",
                    "error": str(e)
                }
                print(f"  ❌ Basic connectivity: {str(e)}")
                continue
            
            # Test 2: Parameter validation
            try:
                await test_agent.call_api("Test", {
                    "max_tokens": 1,        # Minimum
                    "temperature": 0.0,     # Minimum
                    "top_p": 1.0,          # Maximum
                    "top_k": 1             # Minimum
                })
                validation_results["parameter_validation"][server_name] = "✅ Parameters validated"
                print(f"  ✅ Parameter validation: OK")
            except Exception as e:
                validation_results["parameter_validation"][server_name] = f"❌ {str(e)}"
                print(f"  ❌ Parameter validation: {str(e)}")
            
            # Test 3: Response format
            try:
                response = await test_agent.call_api("Generate a number", {"max_tokens": 20})
                if isinstance(response, str) and len(response) > 0:
                    validation_results["response_format"][server_name] = "✅ Valid response format"
                    print(f"  ✅ Response format: OK")
                else:
                    validation_results["response_format"][server_name] = f"❌ Invalid format: {type(response)}"
                    print(f"  ❌ Response format: Invalid")
            except Exception as e:
                validation_results["response_format"][server_name] = f"❌ {str(e)}"
                print(f"  ❌ Response format test: {str(e)}")
            
            # Test 4: Error handling
            try:
                # Test parameter clamping
                await test_agent.call_api("Test", {"max_tokens": 99999})  # Should be clamped
                validation_results["error_handling"][server_name] = "✅ Error handling works"
                print(f"  ✅ Error handling: OK")
            except Exception as e:
                if "validation" in str(e).lower():
                    validation_results["error_handling"][server_name] = "✅ Proper validation errors"
                    print(f"  ✅ Error handling: Proper validation")
                else:
                    validation_results["error_handling"][server_name] = f"❌ Unexpected error: {str(e)}"
                    print(f"  ❌ Error handling: Unexpected error")
            
            # Cleanup
            test_agent.cleanup()
            
        except Exception as e:
            print(f"  ❌ Server test failed: {str(e)}")
    
    # Summary
    print("\n📋 Validation Summary:")
    for test_category, results in validation_results.items():
        print(f"\n{test_category.replace('_', ' ').title()}:")
        for server_name, result in results.items():
            print(f"  {server_name}: {result}")
    
    return validation_results

# ==================== Example Usage ====================

async def example_api_agent_usage():
    """Example of using the API-compatible base agent with proper configuration"""
    
    # Assuming you have an API coordinator
    from .api_coordinator import create_api_coordinator_from_endpoints
    
    gpu_endpoints = {
        2: "http://localhost:8100",  # Your CodeLlama server
        3: "http://localhost:8101"   # Another server if available
    }
    
    coordinator = create_api_coordinator_from_endpoints(gpu_endpoints)
    await coordinator.initialize()
    
    try:
        # Validate integration first
        validation_results = await validate_agent_server_integration(coordinator, gpu_endpoints)
        
        # Create API-based agent with proper configuration
        agent = BaseOpulenceAgent(coordinator, "example_agent", gpu_id=2)  # GPU ID kept for compatibility
        
        # Update API parameters for this specific use case
        agent.update_api_params(
            max_tokens=500,
            temperature=0.3,
            top_p=0.95,
            top_k=50
        )
        
        # Make API calls with proper error handling
        try:
            response1 = await agent.call_api("Explain COBOL programming in simple terms")
            print(f"Response 1: {response1[:200]}...")
        except Exception as e:
            print(f"API call 1 failed: {e}")
        
        try:
            response2 = await agent.call_api("What is data lineage?", {
                "max_tokens": 200,
                "temperature": 0.1  # Override agent default
            })
            print(f"Response 2: {response2[:200]}...")
        except Exception as e:
            print(f"API call 2 failed: {e}")
        
        # Use context manager (for compatibility with existing agent code)
        try:
            async with agent.get_engine_context() as api_context:
                print("Using API context successfully")
                
                # Test with sampling params object
                class SamplingParams:
                    def __init__(self):
                        self.max_tokens = 100
                        self.temperature = 0.5
                        self.top_p = 0.9
                
                sampling_params = SamplingParams()
                async for output in api_context.generate("Summarize the benefits of code documentation", sampling_params):
                    print(f"Context response: {output.outputs[0].text[:100]}...")
        except Exception as e:
            print(f"Context manager test failed: {e}")
        
        # Get final statistics
        stats = agent.get_agent_stats()
        print(f"Final agent stats: {stats}")
        
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_api_agent_usage())