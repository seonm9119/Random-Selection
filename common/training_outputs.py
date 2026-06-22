def create_dataset_batch_best_basename(dataset_name, batch_size):
    return f"{dataset_name}_batch_{batch_size}_best"


def create_best_artifact_basename(training_config):
    return create_dataset_batch_best_basename(training_config["dataset"], training_config["batch_size"])


def use_best_artifact_file_names(training_config):
    artifact_basename = create_best_artifact_basename(training_config)
    training_config["config_file_name"] = f"{artifact_basename}.json"
    training_config["console_log_file_name"] = f"{artifact_basename}.log"
    training_config["checkpoint_file_template"] = f"{artifact_basename}.pt"
    training_config["best_checkpoint_file_name"] = f"{artifact_basename}.pt"
