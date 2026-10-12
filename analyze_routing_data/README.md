# Standalone module to process expert routing data

What I did:
- Started SGLang server with `--enable-return-routed-experts`
- Used `prsdm/Machine-Learning-QA-dataset` and `whiteOUO/Ladder-machine-learning-QA` as example prompts
- Stored raw responses as JSON
- Routed experts were returned as a base64 encoded string

Example response:
```json
{
    "question": "What is the fundamental goal of machine learning?",
    "response": {
        "id": "34928e43e22a458b8547a894bd7202ec",
        "object": "chat.completion",
        "created": 1791704327,
        "model": "inkling-small",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The fundamental goal of machine learning is to enable computers to learn patterns and relationships from data without being explicitly programmed for every specific task. Rather than following rigid, hard-coded rules, a system uses algorithms to identify structures within a dataset so it can make predictions or decisions based on new, unseen information.\n\nThis process generally involves building a model that improves its performance over time as it is exposed to more data, with the ultimate objective of generalizing well to future situations. While specific applications vary\u2014such as recognizing images, predicting stock prices, or processing language\u2014the core aim remains the same: deriving actionable knowledge or automated capabilities directly from experience and observation.",
                    "reasoning_content": null,
                    "tool_calls": null
                },
                "logprobs": null,
                "finish_reason": "stop",
                "matched_stop": 200006
            }
        ],
        "usage": {
            "prompt_tokens": 21,
            "total_tokens": 153,
            "completion_tokens": 132,
            "prompt_tokens_details": null,
            "reasoning_tokens": 131
        },
        "metadata": {
            "weight_version": "default"
        },
        "sglext": {
            "routed_experts": <base64 encoded string>
        }
    }
}
```