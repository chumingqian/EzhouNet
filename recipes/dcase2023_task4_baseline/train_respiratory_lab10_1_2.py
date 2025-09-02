import argparse
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger


from desed_task.utils.encoder import  RespiraHotEncoder
from local.classes_dict import classes_5labels, binary_labels


from desed_task.dataio.datasets_resp_v10_1_1 import RespiraGnnSet
from desed_task.nnet.EzhouNet_v10_1_1 import GraphRespiratory
from local.respira_trainer_v10_1_2 import RespiraSED_lab


# v9-5-6,   重新配置优化器， 使用Lambda lr,  取消使用Multistep lr 的方式

# v9-5-7,   lightning  从 1.9.5 -- 2.2.5；  更新 lr_scheduler_step;
#    chunk_size = 25
#    stride = 5   # 50%  overlap


# v9-6-4, 将边的标签考虑进去训练， 没有加入位置信息；
#                 chunk_size = 25
#                 stride = 5   # 50%  overlap

# v9-6-5, 将边的标签考虑进去训练,   并且使用对边的特征使用顺序的方式；

#  v9-6-6, 将边的标签考虑进去训练,  并且使用对边的特征使用顺序的方式；
#  将位置信息加入到 区间分类中；


#  v9-6-7, 将边的标签考虑进去训练,  并且使用对边的特征使用顺序的方式；
#  将位置信息同时加入到节点和区间分类中；


# v9-7-1;
"""
1.  对节点和区间的置信度损失，从BCE 替换为 软 soft f1  损失；
2. 引入三元组损失， 进一步用于区分 正常节点和异常节点的node embedding,  从而提高节点置信度； 
3.  暂时去除边的分类损失， 保留节点和区间的 focal loss 的损失方式；
4.   区间的定位损失，替换为 smoth l1 loss;
5.  使用 AdamW 以及 余弦退火算法， 一起合作配合。
"""



# 区间的定位损失，将区间的定位损失  保留为 IOU
# v9-7-3;  修改 区间的 初始学习率为 2e-3;
# 并且修改   Node level loss  的权重为0.2；
# 修改 num triplet = 40; 预热 50 epoch;


# v9-7-5; 使用双优化器， 切换为手动优化，
# 节点和 区间的学习率的衰减速度不同， 并且最低值也不同；



# v9-7-7;  去除节点和区间中的 位置信息；
# 将区间的定位损失保留为 IOU;


# v9-7-9;  边属性为顺序结构；
#  调节 weight  param,  从[-20, 20] --> [-1.5, 1.5];

# v9-8-1:
# 确认区间的定位损失保留为IOU; 去除节点和区间中的位置信息；区间学习率仍使用余弦退火的方式；  weight  param,  从[-20, 20] --> [-1.5, 1.5];
#  仅仅对  {"Stridor", "Rhonchi"} 两种类型使用数据增强；



# v9-8-2:
# 确认区间的定位损失保留为IOU; 去除节点和区间中的位置信息；区间学习率仍使用余弦退火的方式；
#  对先验区间标签， 对不同类型的区间， 使用不同的IOU 阈值进行过滤， crackle 阈值最高，  Stridor 的IOU  阈值最低；


# v9-8-6:修改这里interval  区间学习率调整的代码，  将其调整为多步学习率调整，
# 即100epoch  变为 1e-3,  200epoch  之后成为5e-4，300epoch  之后成为1e-4，   之后 一直保持不变



# 10-1-1 对先验引入中心点，  直接预测中心点的偏移量， 以及宽度的偏移量；
# 去除预测先验区间， 开头和起始预测偏移量的方式；


#  10-1-2,  调整定位损失 smooth l1 的参数，


def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    # handle seed  ,设置随机数种子；
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    # 由于数据集中生成节点特征时， 需要用到CNN提取器， 故这里先进行实例化；
    #####  feature extracter and   model definition  ############

    sed_net = GraphRespiratory(**config["net"])#.cuda()
    ##### data prep test ##########,
    encoder = RespiraHotEncoder(
        list(classes_5labels.keys()),
        list(binary_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_fft"],
        frame_hop=config["feats"]["hop_length"],
        frames_per_node=config["data"]["frames_per_node"],
        fs=config["data"]["fs"],
    )

    if not evaluation: # False 时， 启用 test 测试集
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = RespiraGnnSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,

            fs=config["data"]["fs"],
            n_fft=config["feats"]["n_fft"],
            win_len=config["feats"]["n_window"],
            hop_len=config["feats"]["hop_length"],
            f_max=config["feats"]["f_max"],
            n_filters=config["feats"]["n_mels"],

            # node_fea_generator=spec2node_generator,
            train_flag= False,
            audio_aug= False,
            spec_aug= False,
            return_filename=False,
        )

    else:
        #val_data_df = pd.read_csv(config["data"]["valid_tsv"], sep="\t") # 100 份音频， 8127份事件；
        devtest_dataset = None

        #     RespiraGnnSet(
        #     config["data"]["eval_folder_8k"],
        #     val_data_df,
        #     encoder,
        #     return_filename=True
        # )
    test_dataset = devtest_dataset

     #   此时的数据集中， 去除了所有正常事件的标注信息， 即1. 如果该audio 是normal 类型，则其中的所有normal的事件被删除； 2. 如果该audio 是异常类型，则其中标注为正常的事件，同样被删除；
    #  #NOte, 运算量 calulate multiply–accumulate operation (MACs)
    # macs, params = calculate_macs(sed_net, config)
    # print(f"---------------------------------------------------------------")
    # print(f"Total number of multiply–accumulate operation (MACs): {macs}, the netpara {params} \n")

    if test_state_dict is None:
        ##### data prep train valid ##########,  数据准备， 训练和验证
        train_data_df = pd.read_csv(config["data"]["train_tsv"], sep="\t")
        train_dataset  = RespiraGnnSet(
            config["data"]["train_folder_8k"],
            train_data_df,
            encoder, #  输入网络编码器，

            fs=config["data"]["fs"],
            n_fft=config["feats"]["n_fft"],
            win_len=config["feats"]["n_window"],
            hop_len=config["feats"]["hop_length"],
            f_max=config["feats"]["f_max"],
            n_filters=config["feats"]["n_mels"],

            #node_fea_generator=spec2node_generator,
            train_flag = True,  #  训练集，打开用于对异常样本进行数据增强
            audio_aug= True,
            spec_aug= True,
            return_filename= False,
         )

        # 验证集准备；
        val_data_df = pd.read_csv(config["data"]["valid_tsv"], sep="\t") # 100 份音频， 8127份事件；
        valid_dataset = RespiraGnnSet(
            config["data"]["eval_folder_8k"],
            val_data_df,
            encoder,

            fs=config["data"]["fs"],
            n_fft=config["feats"]["n_fft"],
            win_len=config["feats"]["n_window"],
            hop_len=config["feats"]["hop_length"],
            f_max=config["feats"]["f_max"],
            n_filters=config["feats"]["n_mels"],
            train_flag= False,
            return_filename=True,
        )



        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]),
            config["log_dir"].split("/")[-1],
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val/obj_metric",
                    patience=config["training"]["early_stop_patience"],
                    verbose=True,
                    mode="max",
                ),
                ModelCheckpoint(
                    logger.log_dir,
                    monitor="val/obj_metric",
                    save_top_k=1,
                    mode="max",
                    save_last=True,
                ),

            ]
    else:
        train_dataset = None
        valid_dataset = None
        logger = True
        callbacks = None

    respiratory_trainer = RespiraSED_lab(
        config,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        fast_dev_run=fast_dev_run,
        encoder=encoder,
        GraphNet=sed_net,
        evaluation=evaluation,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1      #10k, 1420, 14412 = 25832; 25832/48 =538 个批次，  538 *0.2
        limit_train_batches = 0.03  #2 , bt =48;
        limit_val_batches = 0.05    #2 , bt=12, 2500 +158= 2658;  2658/bt = 220个批次batch, 220 *0.2 =44 个batch ; 得到 # 使用limit_val_batches = 0.2 ，则仅处理 20% 的验证批次，但每个批次将包含 12 个样本（由val_batch_size设置）。
        limit_test_batches = 1.0
        n_epochs = 270
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 1
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    if gpus == "0":
        accelerator = "cpu"
        devices = 1
    elif gpus == "1":
        accelerator = "gpu"
        devices = 1
    else:
        raise NotImplementedError("Multiple GPUs are currently not supported")



    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        reload_dataloaders_every_n_epochs=config["training"]["reload_dataloaders_every_n_epochs"],
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy ="auto" ,    #=config["training"].get("backend"),

        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
        profiler= "simple",
        #profiler = "advanced",
        #profiler = torch_profiler
        #fast_dev_run=10
    )
    if test_state_dict is None:
        #NOte, 计算能耗 start tracking energy consumption
        trainer.fit(respiratory_trainer, ckpt_path=checkpoint_resume)  # 开始调用模型进行训练；
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    # respiratory_trainer.load_state_dict(test_state_dict)
    # trainer.test(respiratory_trainer)


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        # default="./confs/test_lab1.yaml",
        default="./confs/rep_lab9_5.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/2023_baseline",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )

    parser.add_argument(
        "--strong_real",
        action="store_true",
        default=False,
        help="The strong annotations coming from Audioset will be included in the training phase.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='1', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    parser.add_argument(
        "--eval_from_checkpoint", default=None, help="Evaluate the model specified"
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    evaluation = True  # False
    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True


    test_model_state_dict = None
    test_from_checkpoint = args.test_from_checkpoint
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:  #  Note, 为什么验证集上 batch size 被作者设置成1？
        configs["training"]["batch_size_val"] =  12  #1

    test_only = test_from_checkpoint is not None
    # resample_data_generate_durations(configs["data"], test_only, evaluation)
    return configs, args, test_model_state_dict, evaluation


import multiprocessing  as mp

if __name__ == "__main__":

    # prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()
    # mp.set_start_method("spawn")
    # launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
    )



"""
--fast_dev_run  --gpu 1

	val/obj_metric: 0.11472180259500271
	val/weak/student/macro_F1: 0.06900081783533096	  val/weak/teacher/macro_F1: 0.07613814622163773
	val/synth/student/psds1_sed_scores_eval: 0.0	  val/synth/student/intersection_f1_macro: 0.04572098475967175	val/synth/teacher/intersection_f1_macro: 0.0326496530620187
	val/synth/student/event_f1_macro: 0.0001104972375690608	 val/synth/teacher/event_f1_macro: 0.00010198878123406426
"""