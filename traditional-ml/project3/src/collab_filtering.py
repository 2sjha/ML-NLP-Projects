"""
Implements Collaborative Filtering Algoirithm
"""
from collections import defaultdict
import json


class CollabFilter:
    """
    Implements Collaborative Filtering Algoirithm
    """

    def __init__(self, dataset):
        self.mean_user_ratings = defaultdict(float)
        self.movie_rated_users = defaultdict()

        self.train(dataset)
        self.test(dataset)

    def train(self, dataset):
        """
        Calculates mean votes per user and creates a map of movie to users who have rated that movie
        """
        # Init empty list of users for each movie
        for movie in dataset["movies"].keys():
            self.movie_rated_users[movie] = set()

        for user_id, ratings in dataset["ratings"]["train"].items():
            ratings_count = len(ratings)
            self.mean_user_ratings[user_id] = float(0.0)
            for mv_id, mv_rating in ratings.items():
                # Add user to list of users for this movie mv_id
                self.movie_rated_users[mv_id].add(user_id)
                # Add this movie's rating for mean rating
                self.mean_user_ratings[user_id] += mv_rating
            # Divide by count of ratings to calculate mean rating
            self.mean_user_ratings[user_id] /= ratings_count

    def test(self, dataset):
        """
        Calculates error between predicted ratings and actual ratings
        """
        mean_absolute_error = float(0.0)
        root_mean_squared_error = float(0.0)
        count = 0

        # Calculate weights to access them later, instead of
        # calculating it at run time for every test instance
        test_users = defaultdict()
        for user_id, user_ratings in dataset["ratings"]["test"].items():
            test_users[user_id] = user_ratings.keys()

        self.calculate_weights_for_users(dataset, test_users)

        print("actual_rating\tpredicted_rating")
        print("------------------------------------------------------------------")
        for user_id, user_ratings in dataset["ratings"]["test"].items():
            for mv_id, actual_rating in user_ratings.items():
                predicted_rating = self.predict_user_mv_rating(dataset, user_id, mv_id)
                print(str(actual_rating) + "\t\t\t\t" + str(predicted_rating))

                mean_absolute_error += abs(predicted_rating - actual_rating)
                root_mean_squared_error += pow(abs(predicted_rating - actual_rating), 2)
                count += 1
        print("------------------------------------------------------------------\n")

        mean_absolute_error /= count
        root_mean_squared_error /= count
        root_mean_squared_error = pow(root_mean_squared_error, 0.5)
        print("Mean absolute error: " + str(mean_absolute_error))
        print("Root mean squared error: " + str(root_mean_squared_error))

    def calculate_weight_for_two_users(self, dataset, user_i, user_j) -> float:
        """
        Finds common movies rated by user_i and user_j,
        Uses it to calculate weight for user_i and user_j
        """
        common_rated_movies = set(dataset["ratings"]["train"][user_i].keys()) & set(
            dataset["ratings"]["train"][user_j].keys()
        )
        if len(common_rated_movies) == 0:
            return float(0.0)
        else:
            numerator = float(0.0)
            denominator1 = float(0.0)
            denominator2 = float(0.0)
            for mv_id in common_rated_movies:
                numerator += (
                    dataset["ratings"]["train"][user_i][mv_id]
                    - self.mean_user_ratings[user_i]
                ) * (
                    dataset["ratings"]["train"][user_j][mv_id]
                    - self.mean_user_ratings[user_j]
                )

                denominator1 += pow(
                    (
                        dataset["ratings"]["train"][user_i][mv_id]
                        - self.mean_user_ratings[user_i]
                    ),
                    2,
                )
                denominator2 += pow(
                    (
                        dataset["ratings"]["train"][user_j][mv_id]
                        - self.mean_user_ratings[user_j]
                    ),
                    2,
                )

            if denominator1 == 0 or denominator2 == 0:
                return float(0.0)
            else:
                return float(numerator / pow(denominator1 * denominator2, 0.5))

    def calculate_weights_for_users(self, dataset, test_users):
        """
        calculates rating weights for all test users using equation 2
        """
        for user_id, rated_movies in test_users.items():
            wt_users = set()
            user_wt = defaultdict()
            for mv_id in rated_movies:
                for mv_user in self.movie_rated_users[mv_id]:
                    if mv_user != user_id:
                        wt_users.add(mv_user)

            for mv_user in wt_users:
                wt = self.calculate_weight_for_two_users(dataset, user_id, mv_user)
                user_wt[mv_user] = wt

            # Normalize the weights
            wt_sum = float(0.0)
            for user_j_wt in user_wt.values():
                wt_sum += user_j_wt

            if wt_sum != 0:
                for user_j in user_wt:
                    user_wt[user_j] /= wt_sum

            # Writing weights to disk since I encountered a Memory error
            with open("./weights/" + user_id + ".json", "w", encoding="utf-8") as f:
                json.dump(user_wt, f)

    def predict_user_mv_rating(self, dataset, active_user_id, mv_id):
        """
        predicts user vote for a movie using equation 1
        """
        # Start with mean rating for active user
        predicted_rating = self.mean_user_ratings[active_user_id]

        # Load the weights from disk
        with open(
            "./weights/" + active_user_id + ".json", "r", encoding="utf-8"
        ) as user_wt_json:
            weights = json.load(user_wt_json)
            # Calculate weighted sum of other users' ratings
            for curr_user_i, curr_user_wt in weights.items():
                # Only count rating if this curr_user has rated movie mv_id
                if (
                    mv_id in dataset["ratings"]["train"][curr_user_i]
                    and curr_user_wt != 0
                ):
                    predicted_rating += curr_user_wt * (
                        dataset["ratings"]["train"][curr_user_i][mv_id]
                        - self.mean_user_ratings[curr_user_i]
                    )

                return predicted_rating
