import pandas
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from hcve_lib.cv import cross_validate
from hcve_lib.evaluation_functions import compute_classification_metrics_from_result
from hcve_lib.methods.balanced_xgboost import BalancedXGBoostClassifier

# TODO
# def test_balanced_xgboost():
#     data = pandas.read_csv(
#         './hcve_lib/methods/__tests__/pima_indians_diabetes.csv')
#     X = data.drop(columns='y')
#     y = data['y']
#
#     print(
#         compute_classification_metrics_from_result(
#             cross_validate(
#                 X,
#                 y,
#                 lambda: XGBClassifier(
#                     eval_metric='logloss',
#                     use_label_encoder=False,
#                 ),
#                 n_jobs=1,
#             )['scores'])['f1']['mean'])
#
#     print(
#         compute_classification_metrics_from_result(
#             cross_validate(
#                 X,
#                 y,
#                 lambda: BalancedXGBoostClassifier(
#                     eval_metric='logloss',
#                     use_label_encoder=False,
#                 ),
#                 n_jobs=1,
#                 cv=KFold(
#                     n_splits=10,
#                     shuffle=True,
#                 ).split(X, y),
#             )['scores'])['f1']['mean'])
#
#
# # BalancedXGBoostClassifier()
#
# if __name__ == '__main__':
#     test_balanced_xgboost()

