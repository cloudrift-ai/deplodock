"""Unit tests for compose and nginx generation."""

import yaml
import pytest

from deplodock.commands.deploy import generate_compose, generate_nginx_conf


class TestGenerateCompose:
    def test_single_instance(self, sample_config):
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

    def test_multi_instance(self, sample_config_multi):
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

    def test_compose_parses_as_valid_yaml(self, sample_config):
        sample_config["_num_instances"] = 1
        result = generate_compose(sample_config, "/mnt/models", "test-token")
        parsed = yaml.safe_load(result)
        assert "services" in parsed
        assert "vllm_0" in parsed["services"]

    def test_multi_compose_gpu_allocation(self, sample_config_multi):
        result = generate_compose(sample_config_multi, "/mnt/models", "test-token")
        parsed = yaml.safe_load(result)

        # Instance 0 should get GPUs 0-3, instance 1 gets GPUs 4-7
        vllm_0 = parsed["services"]["vllm_0"]
        vllm_1 = parsed["services"]["vllm_1"]

        gpu_0 = vllm_0["deploy"]["resources"]["reservations"]["devices"][0]
        gpu_1 = vllm_1["deploy"]["resources"]["reservations"]["devices"][0]

        assert gpu_0["device_ids"] == ["0", "1", "2", "3"]
        assert gpu_1["device_ids"] == ["4", "5", "6", "7"]

    def test_image_from_config(self, sample_config):
        sample_config["_num_instances"] = 1
        sample_config["backend"]["vllm"]["image"] = "custom/image:v2"
        result = generate_compose(sample_config, "/mnt/models", "token")
        assert "custom/image:v2" in result


class TestGenerateNginxConf:
    def test_basic_structure(self):
        result = generate_nginx_conf(2)
        assert "least_conn" in result
        assert "vllm_0:8000" in result
        assert "vllm_1:8000" in result
        assert "proxy_buffering off" in result
        assert "listen 8080" in result

    def test_server_count(self):
        result = generate_nginx_conf(4)
        assert "vllm_0:8000" in result
        assert "vllm_1:8000" in result
        assert "vllm_2:8000" in result
        assert "vllm_3:8000" in result
        assert "vllm_4:8000" not in result

    def test_proxy_timeouts(self):
        result = generate_nginx_conf(2)
        assert "proxy_connect_timeout 600s" in result
        assert "proxy_send_timeout 600s" in result
        assert "proxy_read_timeout 600s" in result
