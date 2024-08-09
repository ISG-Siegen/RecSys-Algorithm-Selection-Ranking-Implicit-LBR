import argparse

from available_algorithms import recbole_algorithm_names, lenskit_algorithm_names, recpack_algorithm_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Execution Master")
    parser.add_argument('--mode', dest='mode', type=str, required=True)
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--algorithm_name', dest='algorithm_name', type=str, required=True)
    parser.add_argument('--algorithm_config', dest='algorithm_config', type=int, required=True)
    parser.add_argument('--fold', dest='fold', type=int, required=True)
    args = parser.parse_args()

    if args.algorithm_name in recbole_algorithm_names:
        if args.mode == "fit":
            from run_recbole import recbole_fit

            recbole_fit(**vars(args))
        elif args.mode == "predict":
            from run_recbole import recbole_predict

            recbole_predict(**vars(args))
        elif args.mode == "evaluate":
            from run_recbole import recbole_evaluate

            recbole_evaluate(**vars(args))
        else:
            print(f"Mode {args.mode} not found.")
    elif args.algorithm_name in lenskit_algorithm_names:
        if args.mode == "fit":
            from run_lenskit import lenskit_fit

            lenskit_fit(**vars(args))
        elif args.mode == "predict":
            from run_lenskit import lenskit_predict

            lenskit_predict(**vars(args))
        elif args.mode == "evaluate":
            from run_lenskit import lenskit_evaluate

            lenskit_evaluate(**vars(args))
        else:
            print(f"Mode {args.mode} not found.")
    elif args.algorithm_name in recpack_algorithm_names:
        if args.mode == "fit":
            from run_recpack import recpack_fit

            recpack_fit(**vars(args))
        elif args.mode == "predict":
            from run_recpack import recpack_predict

            recpack_predict(**vars(args))
        elif args.mode == "evaluate":
            from run_recpack import recpack_evaluate

            recpack_evaluate(**vars(args))
        else:
            print(f"Mode {args.mode} not found.")
    else:
        print(f"Algorithm {args.algorithm_name} not found.")
