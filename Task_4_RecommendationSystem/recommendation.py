import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class RecommendationSystem:
    def __init__(self):
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
    
    def build_user_item_matrix(self, ratings_dict):
        """
        Build user-item rating matrix from dictionary
        ratings_dict: {user_id: {item_id: rating}}
        """
        all_items = set()
        for user_ratings in ratings_dict.values():
            all_items.update(user_ratings.keys())
        
        items = sorted(list(all_items))
        users = sorted(list(ratings_dict.keys()))
        
        matrix = np.zeros((len(users), len(items)))
        for i, user in enumerate(users):
            for j, item in enumerate(items):
                matrix[i, j] = ratings_dict[user].get(item, 0)
        
        self.user_item_matrix = matrix
        self.users = users
        self.items = items
        return matrix
    
    def compute_similarity(self):
        """
        Compute cosine similarity between users
        """
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        return self.similarity_matrix
    
    def get_user_recommendations(self, user_id, n_recommendations=5):
        """
        Get recommendations for a user using collaborative filtering
        """
        if user_id not in self.users:
            return []
        
        user_idx = self.users.index(user_id)
        user_ratings = self.user_item_matrix[user_idx]
        
        # Find similar users
        similarities = self.similarity_matrix[user_idx]
        similar_users_idx = np.argsort(similarities)[::-1][1:6]
        
        # Aggregate ratings from similar users
        recommendations = {}
        for sim_user_idx in similar_users_idx:
            sim_user_ratings = self.user_item_matrix[sim_user_idx]
            similarity_score = similarities[sim_user_idx]
            
            for item_idx in range(len(self.items)):
                if user_ratings[item_idx] == 0 and sim_user_ratings[item_idx] > 0:
                    if item_idx not in recommendations:
                        recommendations[item_idx] = 0
                    recommendations[item_idx] += sim_user_ratings[item_idx] * similarity_score
        
        # Sort and return top N recommendations
        sorted_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [self.items[item_idx] for item_idx, _ in sorted_items[:n_recommendations]]
        return recommended_items
    
    def get_item_based_recommendations(self, user_id, n_recommendations=5):
        """
        Get recommendations using item-based collaborative filtering
        """
        if user_id not in self.users:
            return []
        
        user_idx = self.users.index(user_id)
        user_ratings = self.user_item_matrix[user_idx]
        
        # Compute item similarity
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        # Find similar items to ones user liked
        recommendations = {}
        for item_idx in np.where(user_ratings > 0)[0]:
            similar_items = np.argsort(item_similarity[item_idx])[::-1][1:]
            
            for similar_item_idx in similar_items:
                if user_ratings[similar_item_idx] == 0:
                    if similar_item_idx not in recommendations:
                        recommendations[similar_item_idx] = 0
                    recommendations[similar_item_idx] += item_similarity[item_idx][similar_item_idx]
        
        sorted_items = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [self.items[item_idx] for item_idx, _ in sorted_items[:n_recommendations]]
        return recommended_items

if __name__ == "__main__":
    # Example usage
    ratings_data = {
        1: {1: 5, 2: 3, 3: 4, 4: 0},
        2: {1: 3, 2: 2, 3: 0, 4: 4},
        3: {1: 4, 2: 3, 3: 5, 4: 0},
        4: {1: 0, 2: 4, 3: 4, 4: 5}
    }
    
    # Initialize and build system
    recommender = RecommendationSystem()
    matrix = recommender.build_user_item_matrix(ratings_data)
    recommender.compute_similarity()
    
    # Get recommendations
    user_id = 1
    recommendations = recommender.get_user_recommendations(user_id, n_recommendations=3)
    print(f"\nUser-based recommendations for user {user_id}: {recommendations}")
    
    item_recommendations = recommender.get_item_based_recommendations(user_id, n_recommendations=3)
    print(f"Item-based recommendations for user {user_id}: {item_recommendations}")
