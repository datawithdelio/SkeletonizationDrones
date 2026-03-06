from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TrajectoryState:
    position: np.ndarray
    velocity: np.ndarray
    heading_deg: float
    speed: float
    predicted_position: np.ndarray


@dataclass
class TrajectoryUncertaintyState(TrajectoryState):
    covariance: np.ndarray
    heading_confidence: float
    speed_confidence: float


@dataclass
class TrajectoryPredictor:
    """
    Constant-velocity predictor for 2D object trajectory.
    """

    horizon_seconds: float = 0.5
    min_points_for_velocity: int = 2
    history: List[Tuple[float, float, float]] = field(default_factory=list)

    def update(self, x: float, y: float, timestamp: float) -> Optional[TrajectoryState]:
        self.history.append((x, y, timestamp))
        if len(self.history) < self.min_points_for_velocity:
            return None

        if len(self.history) > 32:
            self.history = self.history[-32:]

        p1 = self.history[-2]
        p2 = self.history[-1]
        dt = max(1e-6, p2[2] - p1[2])
        velocity = np.array([(p2[0] - p1[0]) / dt, (p2[1] - p1[1]) / dt], dtype=float)
        speed = float(np.linalg.norm(velocity))
        heading_rad = float(np.arctan2(velocity[1], velocity[0]))
        heading_deg = float(np.degrees(heading_rad))
        pos = np.array([p2[0], p2[1]], dtype=float)
        predicted = pos + velocity * self.horizon_seconds

        return TrajectoryState(
            position=pos,
            velocity=velocity,
            heading_deg=heading_deg,
            speed=speed,
            predicted_position=predicted,
        )


@dataclass
class KalmanTrajectoryPredictor:
    """
    2D constant-velocity Kalman filter with uncertainty-aware confidence outputs.

    State: [x, y, vx, vy]
    Measurement: [x, y]
    """

    horizon_seconds: float = 0.5
    process_noise: float = 8.0
    measurement_noise: float = 12.0
    initialized: bool = False
    prev_timestamp: Optional[float] = None
    state: np.ndarray = field(default_factory=lambda: np.zeros((4, 1), dtype=float))
    cov: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=float) * 200.0)

    def _build_matrices(self, dt: float):
        f = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        q = np.array(
            [
                [0.25 * dt**4, 0.0, 0.5 * dt**3, 0.0],
                [0.0, 0.25 * dt**4, 0.0, 0.5 * dt**3],
                [0.5 * dt**3, 0.0, dt**2, 0.0],
                [0.0, 0.5 * dt**3, 0.0, dt**2],
            ],
            dtype=float,
        ) * self.process_noise
        r = np.eye(2, dtype=float) * self.measurement_noise
        return f, h, q, r

    def update(self, x: float, y: float, timestamp: float) -> Optional[TrajectoryUncertaintyState]:
        if not self.initialized:
            self.state = np.array([[x], [y], [0.0], [0.0]], dtype=float)
            self.cov = np.eye(4, dtype=float) * 120.0
            self.prev_timestamp = timestamp
            self.initialized = True
            return None

        dt = max(1e-3, float(timestamp - (self.prev_timestamp or timestamp)))
        self.prev_timestamp = timestamp
        f, h, q, r = self._build_matrices(dt)

        # Predict
        x_pred = f @ self.state
        p_pred = f @ self.cov @ f.T + q

        # Update
        z = np.array([[x], [y]], dtype=float)
        innovation = z - (h @ x_pred)
        s = h @ p_pred @ h.T + r
        k = p_pred @ h.T @ np.linalg.inv(s)

        self.state = x_pred + k @ innovation
        i = np.eye(4, dtype=float)
        self.cov = (i - k @ h) @ p_pred

        position = self.state[:2, 0]
        velocity = self.state[2:, 0]
        speed = float(np.linalg.norm(velocity))
        heading = float(np.degrees(np.arctan2(velocity[1], velocity[0])))
        predicted_position = position + velocity * self.horizon_seconds

        position_uncertainty = float(np.sqrt(max(1e-9, self.cov[0, 0] + self.cov[1, 1])))
        velocity_uncertainty = float(np.sqrt(max(1e-9, self.cov[2, 2] + self.cov[3, 3])))

        heading_confidence = float(np.exp(-velocity_uncertainty / 14.0))
        speed_confidence = float(np.exp(-velocity_uncertainty / 10.0))

        # Penalize confidence at near-zero speed where heading is unstable.
        if speed < 0.2:
            heading_confidence *= 0.5
            speed_confidence *= 0.7

        # Additional confidence scaling by positional certainty.
        certainty_scale = float(np.exp(-position_uncertainty / 45.0))
        heading_confidence *= certainty_scale
        speed_confidence *= certainty_scale

        return TrajectoryUncertaintyState(
            position=position.astype(float),
            velocity=velocity.astype(float),
            heading_deg=heading,
            speed=speed,
            predicted_position=predicted_position.astype(float),
            covariance=self.cov.copy(),
            heading_confidence=max(0.0, min(1.0, heading_confidence)),
            speed_confidence=max(0.0, min(1.0, speed_confidence)),
        )
