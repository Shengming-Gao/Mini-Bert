Pretrain: python3 multitask_classifier.py --option pretrain --lr 1e-3 --use_gpu

Finetune: python3 multitask_classifier.py --option finetune --lr 1e-5  --use_gpu  --pretrain_filepath "file_path" (e.g. pretrain-10-0.001-multitask-20241212_114602.pt)

evalute_model(both pretrain and finetune is ok): python3 evaluate_pretrained_model.py --use_gpu  --filepath "file_path" (e.g.finetune-10-1e-05-multitask-20241213_172357.pt)


You may also customize other hyperparameters like batchsize, number of epochs, hidden_dropout_prob, etc. 
