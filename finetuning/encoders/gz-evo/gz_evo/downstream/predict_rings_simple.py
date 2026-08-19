"""Predict rings test set from checkpoint, save preds + true (aligned by order)."""
import os, logging
import hydra, torch
import pandas as pd
import lightning as L
from lightning.pytorch import loggers

from zoobot.pytorch.training import finetune as zoobot_finetune
from gz_evo.downstream.finetune import (
    prepare_experiment, apply_dataset_specific_overrides, get_results_dir,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    logging.basicConfig(level=logging.INFO)
    L.seed_everything(cfg.seed)
    ckpt_path = cfg.ckpt_path
    assert os.path.exists(ckpt_path), ckpt_path

    cfg.name = f"{cfg.learner.architecture_name}_{cfg.dataset}"
    # one folder per divisor; each seed writes its own file inside (avoids collisions)
    cfg.run_id = f"predict_{cfg.dataset}_div{cfg.divisor}"
    save_dir = os.path.join(get_results_dir(), cfg.version, cfg.run_id)
    os.makedirs(save_dir, exist_ok=True)

    apply_dataset_specific_overrides(cfg)
    model, datamodule = prepare_experiment(cfg, token=None)

    try:
        model = type(model).load_from_checkpoint(ckpt_path)
    except Exception as e:
        logging.warning(f"load_from_checkpoint failed ({e}); state_dict fallback")
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd)

    datamodule.setup("fit")
    trainer = zoobot_finetune.get_trainer(
        save_dir, accelerator=cfg.hardware.accelerator, devices=1, num_nodes=1,
        strategy="auto", precision=cfg.hardware.precision,
        enable_progress_bar=True, logger=loggers.CSVLogger(name=cfg.name, save_dir=save_dir),
    )

    # predictions on test set, in dataloader order
    preds = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    preds = torch.concat(preds).numpy().squeeze()           # predicted ring fraction
    true = datamodule.dataset_dict["test"].to_pandas()["label"].values  # same order

    out = pd.DataFrame({"pred": preds, "true": true})
    out_path = os.path.join(save_dir, f"rings_test_preds_seed{cfg.seed}.csv")
    out.to_csv(out_path, index=False)
    logging.info(f"Saved {len(out)} predictions to {out_path}")


if __name__ == "__main__":
    main()
