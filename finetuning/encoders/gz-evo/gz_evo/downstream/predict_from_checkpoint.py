"""
Run test() + save_predictions() from an existing checkpoint, no retraining.
Reuses the pipeline's own prepare_experiment / save_predictions so transforms,
loss, and output format are identical to a normal run.
"""
import os
import logging

import hydra
import lightning as L
from lightning.pytorch import loggers

from zoobot.pytorch.training import finetune as zoobot_finetune
from gz_evo.downstream.finetune import (
    prepare_experiment,
    save_predictions,
    apply_dataset_specific_overrides,
    get_results_dir,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    L.seed_everything(cfg.seed)

    ckpt_path = cfg.ckpt_path
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"

    cfg.name = f"{cfg.learner.architecture_name}_{cfg.dataset}"
    cfg.run_id = f"predict_{cfg.dataset}_{cfg.seed}"
    save_dir = os.path.join(get_results_dir(), cfg.version, cfg.run_id)
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Saving predictions to {save_dir}")

    apply_dataset_specific_overrides(cfg)

    # build model + datamodule exactly as training would
    model, datamodule = prepare_experiment(cfg, token=None)

    # load YOUR weights instead of training
    logging.info(f"Loading checkpoint: {ckpt_path}")
    try:
        model = type(model).load_from_checkpoint(ckpt_path)
    except Exception as e:
        logging.warning(f"load_from_checkpoint failed ({e}); falling back to state_dict load")
        import torch
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)

    datamodule.setup("fit")

    logger = loggers.CSVLogger(name=cfg.name, save_dir=save_dir)
    trainer = zoobot_finetune.get_trainer(
        save_dir,
        accelerator=cfg.hardware.accelerator,
        devices=1,
        num_nodes=1,
        strategy="auto",
        precision=cfg.hardware.precision,
        enable_progress_bar=True,
        logger=logger,
    )

    trainer.test(model=model, datamodule=datamodule)
    save_predictions(model, datamodule, trainer, save_dir)
    logging.info(f"Done. Check predictions_test.csv in {save_dir}")


if __name__ == "__main__":
    main()
