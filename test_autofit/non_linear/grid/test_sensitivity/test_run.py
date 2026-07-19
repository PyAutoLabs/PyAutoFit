# from autonerves.conf import with_config
#
#
# @with_config("general", "model", value=True)
# def test_sensitivity(sensitivity):
#     results = sensitivity.run()
#     assert len(results) == 8
#
#     output_path = sensitivity.paths.output_path
#
#     assert (output_path / ".is_grid_search").exists()
#     path = output_path / "results.csv"
#     assert path.exists()
#     with open(path) as f:
#         all_lines = set(f)
#
#         assert (
#             "index,centre,normalization,sigma,log_evidence_increase,log_likelihood_increase\n"
#             in all_lines
#         )
#         assert "     0,  0.25,  0.25,  0.25,  None,   1.0\n" in all_lines
#     #    assert "     1,  0.25,  0.25,  0.75,  None,   0.0\n" in all_lines
#
#
# def test_serial(sensitivity):
#     sensitivity.number_of_cores = 1
#
#     results = sensitivity.run()
#     assert len(results) == 8
