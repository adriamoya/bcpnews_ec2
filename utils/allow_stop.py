def possible_stop():
    # Proceed ?
    while True:
        start_crawling = input("Do you wish to proceed? [y/n]: ") # raw_input for python2
        if start_crawling.lower() in ["y", ""]:
            print("")
            break
        elif start_crawling.lower() == "n": # abort the process
            print("\nStopping process ...\n")
            exit()
        else:
            print("\nPlease type y or n.\n")
