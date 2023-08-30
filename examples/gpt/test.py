from transformers import AutoModelForCausalLM


if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained("gpt2",
                                                 device_map="auto",
                                                 trust_remote_code=True)

    # for name, param in model.named_parameters():
    #     print("===> ", name)