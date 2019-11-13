class DifficultySchedule(object):
  def __init__(
      self,
      initial_difficulty:float,
      difficulty_step:float,
      target_performance:float
  ):
    assert 0 <= initial_difficulty <= 1
    assert 0 <= difficulty_step <= 1
    assert 0 <= target_performance <= 1
    self._current_difficulty = initial_difficulty
    self.target_performance = target_performance
    self.difficulty_step = difficulty_step

  def current_difficulty(self):
    return self._current_difficulty

  def update_current_performance(self, current_performance:float):
    if current_performance >= self.target_performance:
      self._current_difficulty = min(
          self._current_difficulty+self.difficulty_step,
          1
      )


