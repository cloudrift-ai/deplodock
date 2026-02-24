"""Unit tests for compose and nginx generation."""

import yaml

from deplodock.deploy import generate_compose, generate_nginx_conf

# ── generate_compose ────────────────────────────────────────────────


def test_compose_single_instance(sample_config):
    sample_config["_num_instances"] = 1
    result = generate_compose(sample_config, "/mnt/models", "test-token")

    assert "vllm_0:" in result
    assert "vllm_1:" not in result
    assert "nginx:" not in result
    assert "count: all" in result
    assert '"8000:8000"' in result
    assert "--tensor-parallel-size 1" in result
    assert "--pipeline-parallel-size 1" in result
    assert "--model test-org/test-model" in result
    assert "--served-model-name test-org/test-model" in result
    assert "--max-model-len 8192" in result
    assert "HUGGING_FACE_HUB_TOKEN=test-token" in result
    assert "/mnt/models:/mnt/models" in result


def test_compose_context_length_and_max_concurrent(sample_config):
    sample_config["_num_instances"] = 1
    sample_config["backend"]["vllm"]["max_concurrent_requests"] = 256
    result = generate_compose(sample_config, "/mnt/models", "token")
    assert "--max-model-len 8192" in result
    assert "--max-num-seqs 256" in result


def test_compose_omits_unset_named_fields():
    config = {
        "model": {"name": "test-org/test-model"},
        "backend": {
            "vllm": {
                "image": "vllm/vllm-openai:latest",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
            }
        },
        "_num_instances": 1,
    }
    result = generate_compose(config, "/mnt/models", "token")
    assert "--max-model-len" not in result
    assert "--max-num-seqs" not in result


def test_compose_multi_instance(sample_config_multi):
    result = generate_compose(sample_config_multi, "/mnt/models", "test-token")

    # Two vLLM services
    assert "vllm_0:" in result
    assert "vllm_1:" in result
    assert "vllm_2:" not in result

    # Nginx load balancer
    assert "nginx:" in result
    assert "nginx_lb" in result
    assert '"8080:8080"' in result

    # GPU device IDs for multi-instance (not count: all)
    assert "device_ids:" in result
    assert "'0'" in result
    assert "'3'" in result

    # Ports
    assert '"8000:8000"' in result
    assert '"8001:8000"' in result


def test_compose_parses_as_valid_yaml(sample_config):
    sample_config["_num_instances"] = 1
    result = generate_compose(sample_config, "/mnt/models", "test-token")
    parsed = yaml.safe_load(result)
    assert "services" in parsed
    assert "vllm_0" in parsed["services"]


def test_compose_multi_gpu_allocation(sample_config_multi):
    result = generate_compose(sample_config_multi, "/mnt/models", "test-token")
    parsed = yaml.safe_load(result)

    # Instance 0 should get GPUs 0-3, instance 1 gets GPUs 4-7
    vllm_0 = parsed["services"]["vllm_0"]
    vllm_1 = parsed["services"]["vllm_1"]

    gpu_0 = vllm_0["deploy"]["resources"]["reservations"]["devices"][0]
    gpu_1 = vllm_1["deploy"]["resources"]["reservations"]["devices"][0]

    assert gpu_0["device_ids"] == ["0", "1", "2", "3"]
    assert gpu_1["device_ids"] == ["4", "5", "6", "7"]


def test_compose_image_from_config(sample_config):
    sample_config["_num_instances"] = 1
    sample_config["backend"]["vllm"]["image"] = "custom/image:v2"
    result = generate_compose(sample_config, "/mnt/models", "token")
    assert "custom/image:v2" in result


def test_compose_gpu_device_ids_override(sample_config):
    """_gpu_device_ids restricts GPU visibility in single-instance mode."""
    sample_config["_num_instances"] = 1
    sample_config["_gpu_device_ids"] = [0, 1]
    result = generate_compose(sample_config, "/mnt/models", "token")
    assert "device_ids:" in result
    assert "'0'" in result
    assert "'1'" in result
    assert "count: all" not in result


def test_compose_no_gpu_device_ids_uses_count_all(sample_config):
    """Without _gpu_device_ids, single-instance uses count: all."""
    sample_config["_num_instances"] = 1
    result = generate_compose(sample_config, "/mnt/models", "token")
    assert "count: all" in result
    assert "device_ids:" not in result


# ── generate_nginx_conf ─────────────────────────────────────────────


def test_nginx_basic_structure():
    result = generate_nginx_conf(2)
    assert "least_conn" in result
    assert "vllm_0:8000" in result
    assert "vllm_1:8000" in result
    assert "proxy_buffering off" in result
    assert "listen 8080" in result


def test_nginx_server_count():
    result = generate_nginx_conf(4)
    assert "vllm_0:8000" in result
    assert "vllm_1:8000" in result
    assert "vllm_2:8000" in result
    assert "vllm_3:8000" in result
    assert "vllm_4:8000" not in result


def test_nginx_proxy_timeouts():
    result = generate_nginx_conf(2)
    assert "proxy_connect_timeout 600s" in result
    assert "proxy_send_timeout 600s" in result
    assert "proxy_read_timeout 600s" in result
