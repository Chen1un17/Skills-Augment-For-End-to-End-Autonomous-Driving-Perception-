from ad_cornercase.providers.openai_responses import _request_overrides


def test_request_overrides_disable_thinking_for_text_qwen3_models() -> None:
    assert _request_overrides("Qwen/Qwen3-8B") == {"enable_thinking": False}


def test_request_overrides_do_not_send_enable_thinking_to_qwen3_vl_models() -> None:
    assert _request_overrides("Qwen/Qwen3-VL-8B-Instruct") == {}


def test_request_overrides_do_not_send_enable_thinking_to_qwen3_omni_models() -> None:
    assert _request_overrides("Qwen/Qwen3-Omni-30B-A3B-Instruct") == {}
