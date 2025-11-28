"""
Load Testing and Performance Benchmarks

These tests validate system performance under load.
Tests are designed to FAIL if performance degrades below acceptable levels.

CRITICAL: If any test fails, optimize the CODEBASE, NOT the tests.
"""

import pytest
import numpy as np
import time
import joblib
import concurrent.futures
from threading import Lock


class TestInferencePerformance:
    """Test model inference performance."""
    
    @pytest.fixture(scope='class')
    def model_components(self):
        """Load model components once for all tests."""
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, scaler, label_encoder
    
    def test_single_prediction_latency(self, model_components):
        """Test latency for single prediction."""
        model, scaler, _ = model_components
        
        # Generate random input (22 features)
        X = np.random.rand(1, 22)
        
        # Warm-up
        for _ in range(10):
            model.predict(X)
        
        # Measure latency
        latencies = []
        for _ in range(100):
            start = time.time()
            model.predict(X)
            latencies.append((time.time() - start) * 1000)  # Convert to ms
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Assert performance targets
        assert mean_latency < 1.0, f"Mean latency {mean_latency:.2f}ms exceeds 1ms target"
        assert p95_latency < 2.0, f"P95 latency {p95_latency:.2f}ms exceeds 2ms target"
        assert p99_latency < 5.0, f"P99 latency {p99_latency:.2f}ms exceeds 5ms target"
    
    def test_batch_prediction_performance(self, model_components):
        """Test batch prediction performance."""
        model, _, _ = model_components
        
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            X = np.random.rand(batch_size, 22)
            
            # Warm-up
            model.predict(X)
            
            # Measure
            start = time.time()
            predictions = model.predict(X)
            elapsed = (time.time() - start) * 1000  # ms
            
            # Validate
            assert len(predictions) == batch_size, f"Wrong number of predictions for batch size {batch_size}"
            
            # Performance target: < 100ms for 1000 samples
            if batch_size == 1000:
                assert elapsed < 100, f"Batch prediction {elapsed:.2f}ms exceeds 100ms target for 1000 samples"
            
            # Calculate throughput
            throughput = batch_size / (elapsed / 1000)  # predictions per second
            print(f"Batch size {batch_size}: {elapsed:.2f}ms, {throughput:.0f} pred/sec")
    
    def test_memory_usage_during_inference(self, model_components):
        """Test memory usage doesn't grow during inference."""
        import psutil
        import os
        
        model, _, _ = model_components
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run many predictions
        for _ in range(1000):
            X = np.random.rand(10, 22)
            model.predict(X)
        
        # Check memory after
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory
        
        # Memory should not grow significantly (allow 10MB growth)
        assert memory_growth < 10, f"Memory grew by {memory_growth:.2f}MB during inference"


class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    @pytest.fixture(scope='class')
    def model_components(self):
        """Load model components."""
        model = joblib.load('models/production_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        return model, scaler, label_encoder
    
    def make_prediction(self, model_components):
        """Helper function to make a single prediction."""
        model, scaler, label_encoder = model_components
        X = np.random.rand(1, 22)
        prediction = model.predict(X)[0]
        return prediction
    
    def test_concurrent_predictions_10(self, model_components):
        """Test 10 concurrent predictions."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.make_prediction, model_components) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 10, "Not all concurrent predictions completed"
        assert all(0 <= r < 22 for r in results), "Invalid prediction values"
    
    def test_concurrent_predictions_100(self, model_components):
        """Test 100 concurrent predictions."""
        start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(self.make_prediction, model_components) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        elapsed = time.time() - start
        
        assert len(results) == 100, "Not all concurrent predictions completed"
        assert all(0 <= r < 22 for r in results), "Invalid prediction values"
        assert elapsed < 5.0, f"100 concurrent predictions took {elapsed:.2f}s (should be < 5s)"
    
    def test_thread_safety(self, model_components):
        """Test that model is thread-safe."""
        model, _, _ = model_components
        results = []
        lock = Lock()
        
        def predict_and_store():
            X = np.random.rand(1, 22)
            pred = model.predict(X)[0]
            with lock:
                results.append(pred)
        
        # Run 50 predictions concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(predict_and_store) for _ in range(50)]
            concurrent.futures.wait(futures)
        
        assert len(results) == 50, "Thread safety issue: lost predictions"
        assert all(0 <= r < 22 for r in results), "Thread safety issue: invalid predictions"


class TestScalability:
    """Test system scalability."""
    
    def test_increasing_load(self):
        """Test performance under increasing load."""
        model = joblib.load('models/production_model.pkl')
        
        load_levels = [10, 50, 100, 500, 1000]
        latencies = []
        
        for load in load_levels:
            X = np.random.rand(load, 22)
            
            start = time.time()
            predictions = model.predict(X)
            elapsed = (time.time() - start) * 1000  # ms
            
            per_sample_latency = elapsed / load
            latencies.append(per_sample_latency)
            
            assert len(predictions) == load, f"Wrong number of predictions for load {load}"
        
        # Latency should not increase significantly with load
        # (indicates good scalability)
        max_latency = max(latencies)
        min_latency = min(latencies)
        latency_ratio = max_latency / min_latency
        
        assert latency_ratio < 2.0, f"Latency increased {latency_ratio:.2f}x with load (poor scalability)"
    
    def test_sustained_load(self):
        """Test performance under sustained load."""
        model = joblib.load('models/production_model.pkl')
        
        # Run predictions for 10 seconds
        start_time = time.time()
        prediction_count = 0
        
        while time.time() - start_time < 10:
            X = np.random.rand(100, 22)
            predictions = model.predict(X)
            prediction_count += len(predictions)
        
        elapsed = time.time() - start_time
        throughput = prediction_count / elapsed
        
        # Should maintain high throughput
        assert throughput > 10000, f"Sustained throughput {throughput:.0f} pred/sec below 10,000 target"


class TestResourceCleanup:
    """Test resource cleanup after operations."""
    
    def test_no_file_handle_leaks(self):
        """Test that file handles are properly closed."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline file handles
        baseline_fds = len(process.open_files())
        
        # Load and unload model multiple times
        for _ in range(10):
            model = joblib.load('models/production_model.pkl')
            X = np.random.rand(10, 22)
            model.predict(X)
            del model
        
        # Check file handles after
        final_fds = len(process.open_files())
        fd_growth = final_fds - baseline_fds
        
        # Should not leak file handles
        assert fd_growth <= 2, f"File handle leak detected: {fd_growth} handles not closed"
    
    def test_memory_cleanup_after_large_batch(self):
        """Test memory is cleaned up after large batch prediction."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        model = joblib.load('models/production_model.pkl')
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Large batch prediction
        X_large = np.random.rand(10000, 22)
        predictions = model.predict(X_large)
        del X_large
        del predictions
        
        # Force garbage collection
        gc.collect()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory
        
        # Memory should return to near baseline (allow 20MB growth)
        assert memory_growth < 20, f"Memory not cleaned up: {memory_growth:.2f}MB growth"


class TestFlaskAppLoad:
    """Test Flask application under load."""
    
    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        from app import create_app
        app = create_app('testing')
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_concurrent_api_requests(self, client):
        """Test concurrent API requests."""
        import json
        
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        def make_request():
            response = client.post('/predict/api/crop',
                                  data=json.dumps(payload),
                                  content_type='application/json')
            return response.status_code
        
        # Make 20 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            status_codes = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(code == 200 for code in status_codes), "Some concurrent requests failed"
    
    def test_api_response_time(self, client):
        """Test API response time."""
        import json
        
        payload = {
            'N': 90,
            'P': 42,
            'K': 43,
            'temperature': 20.87,
            'humidity': 82.00,
            'ph': 6.50,
            'rainfall': 202.93
        }
        
        # Warm-up
        for _ in range(5):
            client.post('/predict/api/crop',
                       data=json.dumps(payload),
                       content_type='application/json')
        
        # Measure response time
        response_times = []
        for _ in range(50):
            start = time.time()
            response = client.post('/predict/api/crop',
                                  data=json.dumps(payload),
                                  content_type='application/json')
            elapsed = (time.time() - start) * 1000  # ms
            response_times.append(elapsed)
            assert response.status_code == 200
        
        mean_response_time = np.mean(response_times)
        p95_response_time = np.percentile(response_times, 95)
        
        # API should respond quickly
        assert mean_response_time < 100, f"Mean API response time {mean_response_time:.2f}ms exceeds 100ms"
        assert p95_response_time < 200, f"P95 API response time {p95_response_time:.2f}ms exceeds 200ms"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
