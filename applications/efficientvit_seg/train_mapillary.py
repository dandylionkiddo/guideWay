
import argparse
import torch

from efficientvit.apps.trainer import RunConfig
from efficientvit.apps.setup import setup_exp_config, setup_data_provider, setup_run_config, init_model
from efficientvit.models.efficientvit.seg import efficientvit_seg_b0
from efficientvit.segcore.data_provider import MapillaryDataProvider
from efficientvit.segcore.trainer import SegTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--path", type=str, required=True)

def main():
    args, opt_args = parser.parse_known_args()
    exp_config = setup_exp_config(args.config, opt_args=opt_args)

    data_provider = setup_data_provider(
        exp_config,
        data_provider_classes=[MapillaryDataProvider],
        is_distributed=True,
    )

    run_config = setup_run_config(exp_config, RunConfig)

    model = efficientvit_seg_b0(dataset="mapillary")
    init_model(model, exp_config["model_init"])

    trainer = SegTrainer(
        path=args.path,
        model=model,
        data_provider=data_provider,
    )

    trainer.prep_for_training(run_config, ema_decay=0.9998, amp="fp16")
    trainer.load_model()
    trainer.train()

if __name__ == "__main__":
    main()
