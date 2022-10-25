from task_adapted_ae_linlayer import TaskAdaptedAeLinlayer as TAAL
from task_adapted_ae_linlayer_subsampled import TaskAdaptedAeLinlayerSampled as TAALs
from task_adapted_ae_convlayer import TaskAdaptedAeConvlayer as TAAC
from task_adapted_ae_convlayer_exp import TaskAdaptedAeConvlayer as TAAC_exp
from unadapted_ae_linlayer import UnadaptedAeLinlayer as UAL
import torch, os, random, numpy as np, argparse
import pickle, code


def train_with_subsampling(taal_dir, subsample_frac):
    if subsample_frac == 0.5:
        epochs = 12
    elif subsample_frac == 0.25:
        epochs = 24
    elif subsample_frac == 0.1:
        epochs = 60
    else:
        epochs = 60
    print(f"Now training with subsample size {subsample_frac}.", flush=True)
    model = TAALs()
    losses = model.train_model(epochs=epochs, traindir = taal_dir, subsample_frac = subsample_frac)
    torch.save(model.state_dict(), "TaskAdaptedAeLinlayer_NORMALIZED_subsample_%s.pt"%subsample_frac)

def train_taal(taal_dir):
    model = TAAL()
    losses = model.train_model(epochs=6, traindir = taal_dir)
    torch.save(model, "TaskAdaptedAeLinlayer_NORMALIZED_model.pt")
    with open("losses_for_taal.pk", "wb") as output_handle:
        pickle.dump(losses, output_handle)

def train_taac(taac_dir):
    model = TAAC()
    losses = model.train_model(epochs=8, traindir = taac_dir)
    torch.save(model, "TaskAdaptedAeConvlayer_NORMALIZED_model.pt")
    with open("losses_for_taac.pk", "wb") as output_handle:
        pickle.dump(losses, output_handle)

def train_taac_exp(taac_dir):
    model = TAAC_exp()
    losses = model.train_model(epochs=7, traindir = taac_dir)
    torch.save(model, "TaskAdaptedAeConvlayer_exp_model.pt")
    with open("losses_for_taac_exp.pk", "wb") as output_handle:
        pickle.dump(losses, output_handle)


def train_ual(ual_dir):
    model = UAL()
    losses = model.train_model(epochs=2, traindir = ual_dir)
    torch.save(model, "UnadaptedAeLinlayer_model.pt")
    with open("losses_for_ual.pk", "wb") as output_handle:
        pickle.dump(losses, output_handle)

def evaluate_model(model_file, testset_dir, use_cpu = False):
    if use_cpu == False:
        model = TAALs()
        model.load_state_dict(torch.load(model_file))
    else:
        model = TAALs()
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
    current_dir = os.getcwd()
    os.chdir(testset_dir)
    xfilenames = [filename for filename in os.listdir() if filename.endswith("xmix.pt")]
    yfilenames = [filename.split("xmix.pt")[0] + "ymix.pt" for filename in xfilenames]
    aa_preds, cat_preds, aa_gt, cat_gt = [], [], [], []
    for i in range(len(xfilenames)):
        x = torch.load(xfilenames[i])
        y = torch.load(yfilenames[i])
        aa_pred, cat_pred = model.predict(x, True)
        aa_preds.append( np.argmax(aa_pred, axis=-1) )
        cat_preds.append( np.rint(cat_pred) )
        aa_gt.append( np.argmax(x, axis=-1))
        cat_gt.append(y)

    cat_preds, cat_gt = np.concatenate(cat_preds), np.concatenate(cat_gt)
    aa_preds, aa_gt = np.vstack(aa_preds), np.vstack(aa_gt)
    reconstruction_accuracy = 1 - np.argwhere(aa_preds != aa_gt).shape[0] / (aa_gt.shape[0] * aa_gt.shape[1])
    prediction_accuracy = 1 - np.argwhere(cat_preds != cat_gt).shape[0] / cat_gt.shape[0]
    print("Reconstruction accuracy: %s"%reconstruction_accuracy)
    print("Prediction accuracy: %s"%prediction_accuracy)

def main():
    parser = argparse.ArgumentParser(description='Train autoencoders on an assigned training set.')
    parser.add_argument("--TAAL", metavar="TAAL", type=str, help="Input a directory containing a "
            "training set for training a task-adapted autoencoder with linear output layer.")
    parser.add_argument("--TAAC", metavar="TAAC", type=str, help="Input a directory containing a "
            "training set for training a task-adapted autoencoder with conv output layer.")
    parser.add_argument("--evalTAAL", metavar="evalTAAL", type=str, help="Evaluate the task-adapted "
            "linear layer autoencoder on a specified test set directory.")
    parser.add_argument("--evalTAAC", metavar="evalTAAC", type=str, help="Evaluate the task-adapted "
            "conv layer autoencoder on a specified test set directory.")
    parser.add_argument("--UAL", metavar="UAL", type=str, help="Input a directory containing a "
            "training set for training an unadapted autoencoder.")
    parser.add_argument("--TAAC_exp", metavar="TAAC_exp", type=str, help="Input a directory containing a "
            "training set for training an expanded convolutional autoencoder output layer.")
    parser.add_argument("--evalTAAC_exp", metavar="evalTAAC_exp", type=str, help="Evaluate the task-adapted "
            "conv_exp layer autoencoder on a specified test set directory.")
    parser.add_argument("--subsample", type=float, help="Subsample when training TAAL.")
    parser.add_argument("--evalSubsample", type=float, help="Evaluate a subsampled model with the indicated "
                              "subsample fraction.")
    
    args = parser.parse_args()
    if args.TAAL is not None:
        if args.subsample is not None:
            train_with_subsampling(args.TAAL, args.subsample)
        else:
            train_taal(args.TAAL)
    elif args.TAAC is not None:
        train_taac(args.TAAC)
    elif args.evalTAAL is not None and args.evalSubsample is None:
        evaluate_model("TaskAdaptedAeLinlayer_NORMALIZED_model.pt", args.evalTAAL)
    elif args.evalTAAC is not None:
        evaluate_model("TaskAdaptedAeConvlayer_model.pt", args.evalTAAC)
    elif args.UAL is not None:
        train_ual(args.UAL)
    elif args.TAAC_exp is not None:
        train_taac_exp(args.TAAC_exp)
    elif args.evalTAAC_exp is not None:
        evaluate_model("TaskAdaptedAeConvlayer_exp_model.pt", args.evalTAAC_exp)
    elif args.evalSubsample is not None and args.evalTAAL is not None:
        evaluate_model(f"TaskAdaptedAeLinlayer_NORMALIZED_subsample_{args.evalSubsample}.pt",
                            args.evalTAAL, use_cpu = True)


if __name__ == "__main__":
    main()
