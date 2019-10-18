import numpy as np



def meta_loop(main, envs, n_iters, train, **kwargs):



    # Train GP
    print("Training model..")
    main.train_model()
   
    if train:
        n_trials = n_tasks * [1]
    else:
        n_trials = n_tasks * [0]

    for meta_iter in range(n_iters):

        print("Meta iteration {}/{}".format(meta_iter+1, n_iters))

        ## Re-Train GP
        print("Training model..")
        main.train_model()

        if kwargs["model_name"] == "MLSVGP":
            main.plot_H_space()
            main.f_values()
            
            




    n_trials = np.int32(n_trials)

    return n_trials
