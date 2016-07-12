import numpy as np
import bisect
import time
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error

rmean = []
submean = []
movie_similarity = []
iuf_list = []
number_ranked = 300


def compute_mean_of_each_user(data):
    print("Computing means...")
    for userrow in data:
        nozerolist = np.nonzero(userrow)[0]
        movlist = np.zeros(len(userrow))
        if len(nozerolist) == 0:
            rmean.append(0)
        else:
            umean = 0
            for mi in nozerolist:
                umean += userrow[mi]
            umean /= len(nozerolist)
            rmean.append(umean)

            for mi in nozerolist:
                movlist[mi] = userrow[mi] - umean

        submean.append(list(movlist))


def compute_mean_of_each_movie(data):
    print("Computing means...")
    for movrow in data:
        nozerolist = np.nonzero(movrow)[0]
        if len(nozerolist) == 0:
            rmean.append(0)
        else:
            sumlist = 0
            for mi in nozerolist:
                sumlist += movrow[mi]
            rmean.append(sumlist/len(nozerolist))


def correlation_based_similarity(data):
    i = 0
    print("Computing similarty...")
    for movrow in data:
        movsim = []
        nozero1 = list(np.nonzero(movrow)[0])
        c = 0
        for movrow2 in data:
            if i != c:
                nozero2 = list(np.nonzero(movrow2)[0])
                numer = denom1 = denom2 = 0
                sameusers = set(nozero1) & set(nozero2)
                for u in sameusers:
                    numer += (movrow[u] - rmean[i]) * (movrow2[u] - rmean[c])
                    denom1 += np.power((movrow[u] - rmean[i]), 2)
                    denom2 += np.power((movrow2[u] - rmean[c]), 2)

                denom1 = np.sqrt(denom1) * np.sqrt(denom2)
                simval = 0
                if denom1 != 0:
                    simval = numer / denom1

                bisect.insort(movsim, (simval, c))
                if len(movsim) >= number_ranked:
                    movsim.pop(0)
            c += 1
        movie_similarity.append(movsim)
        i += 1


def compute_model_base_prediction(userlist, movie):
    numer = denom = i = 0

    # Computes Weighted Sum of item based collaborative filtering algorithm
    for ms in movie_similarity[movie-1]:
        if userlist[ms[1]] > 0:
            numer += ms[0] * userlist[ms[1]]
            denom += np.abs(ms[0])
        i += 1

    if denom == 0:
        return None
    else:
        return numer / denom


def mem_coll_filter_algo(data, user, movie, usersimilarity):
    numer = denom = i = 0
    for userweight in submean:
        if usersimilarity[i] != 0 and userweight[movie-1] != 0 and i != (user-1):
            numer += usersimilarity[i] * userweight[movie-1]
            denom += np.abs(usersimilarity[i])
        i += 1

    if denom == 0:
        return None
    else:
        return rmean[user-1] + (numer / denom)


# Computes Vector Space Similarity among the given user and other users. Incorporates inverse user frequency
def compute_vector_similarity(data, userlist):
    usersimilarity = []
    userlistdenom = np.sqrt(np.sum(np.power(userlist, 2)))
    for userrow in data:
        usersimilarity.append((np.sum(userrow * userlist)) / (userlistdenom * np.sqrt(np.sum(np.power(userrow, 2)))))
    return usersimilarity


def compute_inverse_user_frequency(tdata):
    number_of_users = np.shape(tdata)[1]
    for itemrow in tdata:
        f = 0
        number_of_users_voted = np.size(np.nonzero(itemrow))
        if number_of_users_voted == 0:
            f = 1
        elif number_of_users_voted != number_of_users:
            f = np.log(number_of_users/number_of_users_voted)

        iuf_list.append(f)


# Computes Vector Space Similarity among the given user and other users. Incorporates inverse user frequency
def compute_vector_similarity_iuf(data, userlist):
    usersimilarity = []
    denom2 = u = 0
    denom2_computed = False
    for userrow in data:
        i = 0
        numer = denom1 = 0
        for mov in userrow:
            iuf = iuf_list[i]
            numer += (mov * iuf) * (userlist[i] * iuf)
            if mov != 0:
                denom1 += np.power((mov * iuf), 2)

            if denom2_computed is False and userlist[i] != 0:
                denom2 += np.power((userlist[i] * iuf), 2)

            i += 1

        if denom2_computed is False:
            denom2 = np.sqrt(denom2)

        denom1 = np.sqrt(denom1) * denom2
        denom2_computed = True
        if denom1 != 0:
            usersimilarity.append(numer/denom1)
        else:
            usersimilarity.append(0)
        u += 1
    return usersimilarity


if __name__ == "__main__":
    inputdata = np.loadtxt("train.txt")
    foldval = 10
    inp = ""
    kf = None
    while inp.lower() != "exit" and inp.lower() != "7":
        inp = input("Enter 1 to execute Memory-based Collaborative Filtering Algorithm\n"
                    "Enter 2 to execute Model-based Collaborative Filtering Algorithm\n"
                    "Enter 3 to execute Memory-based Collaborative Filtering Algorithm with inverse user frequency\n"
                    "Enter 4 to analyze Memory-based Collaborative Filtering Algorithm\n"
                    "Enter 5 to analyze Model-based Collaborative Filtering Algorithm\n"
                    "Enter 6 to analyze Memory-based Collaborative Filtering Algorithm with inverse user frequency\n"
                    "Enter 7 or 'exit' to exit:\n")

        if inp == "1" or inp == "2" or inp == '3':
            while True:
                if (inp == "1" or inp == "3") and len(submean) == 0:
                    del movie_similarity[:]
                    del rmean[:]
                    del submean[:]
                    if inp == "1":
                        del iuf_list[:]
                    elif inp == "3" and len(iuf_list) == 0:
                        compute_inverse_user_frequency(np.transpose(inputdata))

                    compute_mean_of_each_user(inputdata)
                elif inp == "2" and len(movie_similarity) == 0:
                    del movie_similarity[:]
                    del rmean[:]
                    del submean[:]
                    del iuf_list[:]
                    transpose_traindata = np.transpose(inputdata)
                    compute_mean_of_each_movie(transpose_traindata)
                    correlation_based_similarity(transpose_traindata)

                memmovie = memuser = ""
                while True:
                    memuser = input("Enter User: ")
                    if not memuser.isdigit() and memuser != "exit":
                        print("User input has to be a number")
                    elif memuser.isdigit() and int(memuser) < 1:
                        print("No user found (has to be > 0)")
                    else:
                        break
                if memuser == "exit":
                    del movie_similarity[:]
                    del rmean[:]
                    del submean[:]
                    del iuf_list[:]
                    break
                while True:
                    memmovie = input("Enter Movie: ")
                    if not memmovie.isdigit() and memmovie != "exit":
                        print("Movie input has to be a number")
                    elif memmovie.isdigit() and int(memmovie) < 1:
                        print("No movie found (has to be > 0)")
                    else:
                        break
                if memmovie == "exit":
                    del movie_similarity[:]
                    del rmean[:]
                    del submean[:]
                    del iuf_list[:]
                    break

                memmovie = int(memmovie)
                memuser = int(memuser)
                moviepred = None
                if inp == '1':
                    usersimilarity = compute_vector_similarity(inputdata, inputdata[memuser-1])
                    moviepred = mem_coll_filter_algo(inputdata, memuser, memmovie, usersimilarity)
                elif inp == '2':
                    moviepred = compute_model_base_prediction(inputdata[memuser-1], memmovie)
                elif inp == '3':
                    if len(iuf_list) == 0:
                        compute_inverse_user_frequency(np.transpose(inputdata))

                    usersimilarity = compute_vector_similarity_iuf(inputdata, inputdata[memuser-1])
                    moviepred = mem_coll_filter_algo(inputdata, memuser, memmovie, usersimilarity)

                if moviepred is not None:
                    print("Prediction rating for movie: {} for user: {} is: {}\n".format(memmovie, memuser, moviepred))
                else:
                    print("Not enough similar data to compute prediction\n")
        elif inp == "4" or inp == "5" or inp == '6':
            if kf is None:
                kf = KFold(np.shape(inputdata)[0], n_folds=foldval, shuffle=True)

            mae_value = 0
            tstart = time.clock()
            if inp == "4" or inp == "6":
                compute_mean_of_each_user(inputdata)
                if inp == "6":
                    compute_inverse_user_frequency(np.transpose(inputdata))
            elif inp == "5":
                transpose_traindata = np.transpose(inputdata)
                compute_mean_of_each_movie(transpose_traindata)
                correlation_based_similarity(transpose_traindata)

            print("Applying {} Fold Cross Validation...".format(foldval))
            iter = 0
            for train_ind, test_ind in kf:
                # traindata = [inputdata[i] for i in train_ind]
                testdata = [inputdata[i] for i in test_ind]

                mae_mean = 0
                u = num_of_users = 0
                for user in testdata:
                    m = 0
                    predicted_results = []
                    real_value = []
                    for movierating in user:
                        if movierating != 0:
                            moviepred = None
                            if inp == "4" or inp == "6":
                                usersimilarity = []
                                if inp == "6":
                                    usersimilarity = compute_vector_similarity_iuf(inputdata, user)
                                else:
                                    usersimilarity = compute_vector_similarity(inputdata, user)
                                moviepred = mem_coll_filter_algo(inputdata, test_ind[u], m, usersimilarity)
                            elif inp == "5":
                                moviepred = compute_model_base_prediction(user, m)

                            if moviepred is not None:
                                real_value.append(movierating)
                                predicted_results.append(round(moviepred, 7))
                        m += 1

                    if len(real_value) > 0 and len(predicted_results) > 0:
                        mae_mean += mean_absolute_error(real_value, predicted_results)
                        num_of_users += 1
                    u += 1
                mae = 0
                if num_of_users != 0:
                    mae = mae_mean / num_of_users
                    mae_value += mae
                print("Iteration: {} \tMean Abosolute Error: {}".format(iter, mae))
                iter += 1

            if inp == "4":
                print("Memory-based Collaborative Filtering Algorithm:")
            elif inp == "5":
                print("Model-based Collaborative Filtering Algorithm:")
            elif inp == "6":
                print("Memory-based Collaborative Filtering Algorithm with inverse user frequency:")
            print("Execution time: {}".format(time.clock() - tstart))
            print("Final Mean Abosolute Error: {}\n".format(mae_value/foldval))

            del movie_similarity[:]
            del rmean[:]
            del submean[:]
            del iuf_list[:]
