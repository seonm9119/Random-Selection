import json


def reset_training_log(run_dir, training_config):
    training_log_path = run_dir / training_config["train_log_file_name"]
    training_log_path.write_text("", encoding=training_config["text_encoding"])


def write_training_log(run_dir, training_config, training_log):
    training_log_path = run_dir / training_config["train_log_file_name"]
    log_text = json.dumps(training_log, sort_keys=training_config["json_sort_keys"])

    with training_log_path.open("a", encoding=training_config["text_encoding"]) as log_file:
        log_file.write(log_text)
        log_file.write("\n")
