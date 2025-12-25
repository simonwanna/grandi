/*
  Scheduled Monitoring Query
  Action: Calculates metrics for the PREVIOUS DAY's games and updates the model_performance table.
  Using MERGE to ensure idempotency (safe to re-run).
*/

MERGE `chess_monitoring.model_performance` T
USING (
  WITH parsed_games AS (
    -- Step 1: Parse and filter raw game data for the target date
    SELECT
      game_id,
      TIMESTAMP(timestamp) as played_at,
      winner,
      CASE 
        WHEN winner = 'white' THEN 1.0 
        WHEN winner = 'black' THEN 0.0 
        ELSE NULL -- Exclude Draws
      END as target,
      -- Fix single quotes to double quotes for valid JSON
      JSON_EXTRACT_ARRAY(REPLACE(intuition_history, "'", '"')) as history_array
    FROM
      `chess_data.games`
    WHERE
      -- Look at Yesterday's games or Today's
      DATE(TIMESTAMP(timestamp)) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
      -- DATE(TIMESTAMP(timestamp)) = CURRENT_DATE()
  ),
  unnested_moves AS (
    -- Step 2: Unnest game history to analyze individual move predictions
    SELECT
      game_id,
      played_at,
      target,
      CAST(
        COALESCE(
          JSON_VALUE(move_entry, '$[1]'),
          JSON_VALUE(move_entry, '$[1].white'),
          JSON_VALUE(move_entry, '$[1].win_prob')
        ) AS FLOAT64
      ) as predicted_prob_white,
      OFFSET as step_index,
      ARRAY_LENGTH(history_array) as total_steps
    FROM
      parsed_games,
      UNNEST(history_array) as move_entry WITH OFFSET
  ),
  scored_moves AS (
    -- Step 3: Assign game phases and calculate error metrics
    SELECT
      *,
      CASE
        -- Opening: First 25% of moves
        WHEN SAFE_DIVIDE(step_index, total_steps) < 0.25 THEN 'OPENING'
        -- Endgame: Last 25% of moves
        WHEN SAFE_DIVIDE(step_index, total_steps) >= 0.75 THEN 'ENDGAME'
        -- Midgame: Middle 50%
        ELSE 'MIDGAME'
      END as game_phase,
      POW(predicted_prob_white - target, 2) as squared_error
    FROM
      unnested_moves
    WHERE
      predicted_prob_white IS NOT NULL
      AND target IS NOT NULL
  )
  -- Step 4: Aggregate final metrics by Date and Phase
  SELECT
    DATE(played_at) as report_date,
    game_phase,
    COUNT(*) as total_predictions,
    ROUND(AVG(squared_error), 4) as mse,
    ROUND(COUNTIF(
      (predicted_prob_white > 0.5 AND target = 1.0) OR 
      (predicted_prob_white < 0.5 AND target = 0.0) 
    ) / COUNT(*), 4) as rough_accuracy
  FROM
    scored_moves
  GROUP BY
    report_date, game_phase
) S
ON T.report_date = S.report_date AND T.game_phase = S.game_phase
-- Step 5: Upsert results into destination table
WHEN MATCHED THEN
  UPDATE SET 
    total_predictions = S.total_predictions,
    mse = S.mse,
    rough_accuracy = S.rough_accuracy
WHEN NOT MATCHED THEN
  INSERT (report_date, game_phase, total_predictions, mse, rough_accuracy)
  VALUES (S.report_date, S.game_phase, S.total_predictions, S.mse, S.rough_accuracy);
