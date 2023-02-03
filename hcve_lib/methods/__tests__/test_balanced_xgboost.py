# TODO
# def test_balanced_xgboost():
#     data = pandas.read_csv(
#         './hcve_lib/methods/__tests__/pima_indians_diabetes.csv')
#     X = data.drop(columns='y')
#     y = data['y']
#
#     print(
#         compute_classification_metrics_from_result(
#             cross_validate_single_repeat_(
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
#             cross_validate_single_repeat_(
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
