from visualize import visualize_example_data  # noqa: F401
from train_notebook import train_notebook_model  # noqa: F401
from submission import create_submission_for_first_model  # noqa: F401


# TODO: set this up as a CLI switch
def main():
    # train_notebook_model()

    # visualize_example_data()

    create_submission_for_first_model()


if __name__ == "__main__":
    main()
