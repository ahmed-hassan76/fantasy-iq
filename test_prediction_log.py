import unittest
from unittest.mock import patch
import pandas as pd
from src.prediction_log import (build_prediction_log_export, infer_prediction_target_gameweek,
                                complete_prediction_log, validate_snapshot, ActualPointsUnavailable)


class PredictionLogTests(unittest.TestCase):
    def snapshot(self):
        return pd.DataFrame({"Gameweek": ["4", "4"], "Latest Available Source Round": ["3", "2"],
            "Predicted Points": ["2.3400", "-1.20"], "FPL Player ID": ["10", "20"],
            "Export Timestamp": ["2026-08-15T12:00:00+00:00"] * 2, "Notes": ["keep", "unchanged"]})

    def bootstrap(self):
        return {"events": [{"id": 4, "finished": True, "data_checked": True,
                "deadline_time": "2026-09-01T12:00:00Z"}],
                "elements": [{"id": 10}, {"id": 20}], "teams": []}

    def test_target_ignores_historical_labels_and_round(self):
        frame = pd.DataFrame({"source_round": [4, 3], "round": [4, 4], "target_gameweek": [4, 4]})
        self.assertEqual(infer_prediction_target_gameweek(frame), 5)
        with self.assertRaises(ValueError):
            build_prediction_log_export(frame, 4)
        exported, gw = build_prediction_log_export(frame.assign(player_id=[10, 20]), 5)
        self.assertEqual(gw, 5)
        self.assertEqual(exported["FPL Player ID"].tolist(), [10, 20])

    def test_filtered_rows_keep_global_target(self):
        frame = pd.DataFrame({"source_round": [2]})
        frame.attrs["latest_source_round"] = 4
        self.assertEqual(build_prediction_log_export(frame, 5)[1], 5)

    def test_invalid_sources_and_end_of_season(self):
        for value in [None, -1, 4.5, 38, float('inf')]:
            self.assertIsNone(infer_prediction_target_gameweek(pd.DataFrame({"source_round": [value]})))
        for column, value in [("Gameweek", "5"), ("Latest Available Source Round", "4"), ("Latest Available Source Round", ""), ("Predicted Points", "nan")]:
            frame = self.snapshot()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_snapshot(frame)

    @patch('src.prediction_log.fetch_gameweek_live')
    @patch('src.prediction_log.fetch_bootstrap_static')
    def test_completion_preserves_original_fields(self, bootstrap, live):
        bootstrap.return_value = self.bootstrap()
        live.return_value = {"elements": [{"id": 10, "stats": {"total_points": 8}}, {"id": 20, "stats": {"total_points": 0}}]}
        original = self.snapshot()
        result, gw, missing = complete_prediction_log(original)
        pd.testing.assert_frame_equal(result[original.columns], original)
        self.assertEqual((gw, missing), (4, 0))
        self.assertAlmostEqual(result.loc[0, "Error (Actual - Predicted)"], 5.66)
        self.assertAlmostEqual(result.loc[0, "Squared Error"], 5.66 ** 2)
        self.assertAlmostEqual(result.loc[1, "Absolute Error"], 1.2)
        bootstrap.assert_called_once_with(use_cache=False)

    @patch('src.prediction_log.fetch_gameweek_live')
    @patch('src.prediction_log.fetch_bootstrap_static')
    def test_unfinished_or_unchecked_never_fetches_points(self, bootstrap, live):
        for flag in ["finished", "data_checked"]:
            payload = self.bootstrap()
            payload["events"][0][flag] = False
            bootstrap.return_value = payload
            with self.assertRaises(ActualPointsUnavailable):
                complete_prediction_log(self.snapshot())
        live.assert_not_called()

    @patch('src.prediction_log.fetch_gameweek_live')
    @patch('src.prediction_log.fetch_bootstrap_static')
    def test_late_and_wrong_season_exports_rejected(self, bootstrap, live):
        bootstrap.return_value = self.bootstrap()
        for stamp in ["2026-09-02T00:00:00Z", "2025-08-15T00:00:00Z", ""]:
            frame = self.snapshot()
            frame["Export Timestamp"] = stamp
            with self.assertRaises(ValueError):
                complete_prediction_log(frame)
        live.assert_not_called()

    @patch('src.prediction_log.fetch_gameweek_live')
    @patch('src.prediction_log.fetch_bootstrap_static')
    def test_legacy_unique_matching_and_ambiguous_rows(self, bootstrap, live):
        payload = self.bootstrap()
        payload['teams'] = [{"id": 1, "name": "Team"}]
        payload['elements'] = [{"id": pid, "first_name": name, "second_name": "Player", "team": 1, "element_type": 3} for pid, name in [(10, "Unique"), (20, "Same"), (30, "Same")]]
        bootstrap.return_value = payload
        live.return_value = {"elements": [{"id": 10, "stats": {"total_points": 0}}, {"id": 20, "stats": {"total_points": 5}}]}
        frame = self.snapshot().drop(columns='FPL Player ID')
        frame['Player Name'] = ['Unique Player', 'Same Player']
        frame['Team'] = 'Team'
        frame['Position'] = 'MID'
        result, _, missing = complete_prediction_log(frame)
        self.assertEqual(missing, 1)
        self.assertEqual(result.loc[0, 'Actual Points'], 0)
        self.assertTrue(pd.isna(result.loc[1, 'Actual Points']))
        frame['FPL Player ID'] = ['999', '20']
        result, _, missing = complete_prediction_log(frame)
        self.assertTrue(pd.isna(result.loc[0, 'Actual Points']))

    @patch('src.prediction_log.fetch_gameweek_live')
    @patch('src.prediction_log.fetch_bootstrap_static')
    def test_missing_official_points_remain_blank(self, bootstrap, live):
        bootstrap.return_value = self.bootstrap()
        live.return_value = {"elements": [{"id": 10, "stats": {"total_points": 3}}]}
        frame = self.snapshot()
        frame['Actual Points'] = ['99', '99']
        result, _, missing = complete_prediction_log(frame)
        self.assertEqual(missing, 1)
        self.assertTrue(pd.isna(result.loc[1, 'Actual Points']))
        self.assertEqual(result['Predicted Points'].tolist(), frame['Predicted Points'].tolist())
        live.return_value = {"elements": []}
        with self.assertRaises(ActualPointsUnavailable):
            complete_prediction_log(frame)

    def test_id_passthrough_is_not_a_model_feature(self):
        from src.features import split_position_datasets
        from src.constants import FEATURES_BY_POSITION
        frame = pd.DataFrame({'name': ['Player'], 'position': ['MID'], 'player_id': [10], 'round': [3]})
        tables = split_position_datasets(frame, verbose=False)
        self.assertEqual(tables['MID']['player_id'].tolist(), [10])
        for features in FEATURES_BY_POSITION.values():
            self.assertNotIn('player_id', features)

    @patch('src.api._get_json')
    def test_official_points_endpoint_has_no_cache_fallback(self, get_json):
        from src.api import fetch_gameweek_live
        get_json.return_value = {'elements': []}
        fetch_gameweek_live(4)
        get_json.assert_called_once_with('https://fantasy.premierleague.com/api/event/4/live/', use_cache=False)


if __name__ == '__main__':
    unittest.main()
