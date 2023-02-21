from procedures import training_procedure, minimization_procedure


if __name__ == "__main__":
    # table properties
    a = 2
    b = 1

    # magnetic properties
    mu = 1/5

    # training_procedure()
    minimization_procedure("C:/Users/philipp/Documents/documents/uni/bachelor/8/bachelorarbeit/InverseMagneticBilliards/output/checkpoints/model/2023-02-16/model.pth")

    # generate_dataset(a, b, mu, 50000, "train50k.npy")
