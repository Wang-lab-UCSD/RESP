from ..model_code.task_adapted_autoencoder import TaskAdaptedAutoencoder as TAAL
import torch, os, random, numpy as np, argparse
import pickle, code

def train_taal(taal_dir):
    model = TAAL()
    losses = model.train_model(epochs=6, traindir = taal_dir)
    torch.save(model, "TaskAdapted_Autoencoder.ptc")
    with open("losses_for_taal.pk", "wb") as output_handle:
        pickle.dump(losses, output_handle)


def evaluate_model(model_file, testset_dir, use_cpu = False):
    if use_cpu == False:
        model = torch.load(model_file, map_location=torch.device('cuda'))
    else:
        model = torch.load(model_file, map_location=torch.device('cpu'))
    current_dir = os.getcwd()
    os.chdir(testset_dir)
    xfilenames = [filename for filename in os.listdir() if filename.endswith("xmix.pt")]
    yfilenames = [filename.split("xmix.pt")[0] + "ymix.pt" for filename in xfilenames]
    reconstruction_accuracies, prediction_accuracies = [], []
    for i in range(len(xfilenames)):
        x = torch.load(xfilenames[i])
        y = torch.load(yfilenames[i])
        prediction_accuracies.append(model.cat_accuracy(x, y, use_cpu))
        reconstruction_accuracies.append(model.reconstruct_accuracy(x, use_cpu))
    print("Reconstruction accuracy: %s"%reconstruction_accuracies)
    print("Prediction accuracy: %s"%prediction_accuracies)

def main():
    parser = argparse.ArgumentParser(description='Train autoencoder on an assigned training set.')
    parser.add_argument("--TAAL", metavar="TAAL", type=str, help="Input a directory containing a "
            "training set for training a task-adapted autoencoder.")
    parser.add_argument("--evalTAAL", metavar="evalTAAL", type=str, help="Evaluate the task-adapted "
            "autoencoder on a specified test set directory.")
    args = parser.parse_args()
    if args.TAAL is not None:
        train_taal(args.TAAL)
    elif args.evalTAAL is not None:
        evaluate_model("TaskAdapted_Autoencoder.ptc", args.evalTAAL)

if __name__ == "__main__":
    main()
