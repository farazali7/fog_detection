from config import cfg
from main import train_loso
import logging
import optuna as opt
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils
import joblib


def objective(trial):
    locations_drop = []
    #Data dependant 
    #number of modalities (5 different modalities)
    modalities_list = []

    if trial.suggest_int('EMG', 0, 1) == 1:
        modalities_list.append("EMG")
    
    if trial.suggest_int('ECG', 0, 1) == 1:
       modalities_list.append("ECG")

    acc = trial.suggest_int('ACC', 0, 1)
    if acc == 1:
        modalities_list.append("ACC")

    gyr = trial.suggest_int('GYR', 0, 1)
    if gyr == 1:
        modalities_list.append("GYR")

    if gyr == 1 or acc == 1:
        rs = trial.suggest_int('RS', 0, 1)
        ls = trial.suggest_int('LS', 0, 1)
        w = trial.suggest_int('W', 0, 1)
        a = trial.suggest_int('A', 0, 1)

        locations_drop = []
        if rs == 0:
            locations_drop.append("RS-")
        if ls == 0:
            locations_drop.append("LS-")
        if w == 0:
            locations_drop.append("W-")
        if a == 0:
            locations_drop.append("A-")

    if trial.suggest_int('NC/SC', 0, 1) == 1:
        modalities_list.append("NC/SC")
    
    cfg['MODALITIES'] = modalities_list

    # #win.len (3.2, 6.4, 12.8)  - number of samples would be a power to 2
    window_len = trial.suggest_categorical('window_length', [6.4, 12.8])
    cfg['WIN_LENGTH'] = window_len

    #sample rate (Start with 40, but let try higher as well)
    cfg['SAMPLE_RATE'] = trial.suggest_int("Sample_Rate", 40, 200, step=20)

    # Data loading independant

    #Transformers only
    #number of heads **has to be even(2, 4, 8, 16)
    # heads_num_exp = trial.suggest_int('number of heads', 1, 4, step=1)
    # heads_num = 2**heads_num_exp
    # cfg['N_HEADS'] = heads_num

    #number of encoder layers (1-12)
    # encoder_layer_num = trial.suggest_int('number of encoder layers', 1, 12, step=1)
    # cfg['N_ENC_LAYERS'] = encoder_layer_num

    #number of windows (1 - 5)
    windows_num = trial.suggest_int('num_windows', 1, 5, step=1)
    cfg['N_WINDOWS']  = windows_num

    #percentage overlap(0, 0.25, 0.50, 0.75)
    percent_overlap = trial.suggest_float('percentage overlap', 0.50, 0.75, step=0.25)
    cfg['OVERLAP'] =  percent_overlap

    

    #number of filters
    cfg['N_MAX_CONV_FILTERS'] = trial.suggest_int("num_conv_filters", 128, 512, step=64)

    cfg['LR'] = trial.suggest_float("learning_rate", 10**(-4), 0.1, log=True)

    cfg['DROPOUT'] = trial.suggest_float("dropout", 0, 0.5, step=0.1)

    #Start with constant values for batch size, learning rate, betas, epsilon, weight decay
    
    print(cfg)
    train_mean_acc, val_mean_acc, train_mean_auc, val_mean_auc = train_loso(subjects=cfg['SUBJECTS'], modalities=cfg['MODALITIES'], sample_rate=cfg['SAMPLE_RATE'], win_len=cfg['WIN_LENGTH'], overlap=cfg['OVERLAP'], n_windows=cfg['N_WINDOWS'], num_head=cfg['N_HEADS'], num_ec_layers=cfg['N_ENC_LAYERS'], num_filters=cfg['N_MAX_CONV_FILTERS'], 
                                            locations_drop=locations_drop, run=run
                                            )
    
    print("mean train acc: ", train_mean_acc)
    print("mean val acc", val_mean_acc)
    print("mean train auc: ", train_mean_auc)
    print("mean val auc", val_mean_auc)
    print("logged")
    # if trial.should_prune():
    #         raise opt.TrialPruned()
    return val_mean_auc

# #Load previous study?
# experiment_one = joblib.load("study.pkl")

# #Add stream handler to show of trails were prunned. 
# opt.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

run = neptune.init(
    project="SYDE599",
	api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZWUyZjYwYi1lZGE3LTQ2NGUtYTliOC03OWEzZTRlYjRiMDYifQ=="
)

neptune_callback = optuna_utils.NeptuneCallback(run)

study = opt.create_study(study_name='SYDE599_Run', direction='maximize')
study.optimize(objective, n_trials=20, callbacks=[neptune_callback])
print(" Value: ", study.best_trial.value)

# joblib.dump(study, "study.pkl")

# optuna_utils.log_study_metadata(
#     study,
#     run
# )

# To resume study
# study = joblib.load("study.pkl")


# experiment_one = opt.create_study(direction='maximize', pruner=opt.pruners.MedianPruner())


# joblib.dump(experiment_one, "study.pkl")