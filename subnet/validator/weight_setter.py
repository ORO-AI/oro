"""Background thread for periodic weight updates from Backend leaderboard."""

import threading
from typing import Optional

from bittensor.utils.btlogging import logging

from .backend_client import BackendClient, BackendError


class WeightSetterThread:
    """Periodically fetches top miner from Backend and updates on-chain weights.

    Runs in a background thread, independent of evaluation loop.
    Default interval is 5 minutes (300 seconds).
    """

    def __init__(
        self,
        backend_client: BackendClient,
        subtensor,
        metagraph,
        wallet,
        netuid: int,
        interval_seconds: int = 300,
    ):
        self.backend_client = backend_client
        self.subtensor = subtensor
        self.metagraph = metagraph
        self.wallet = wallet
        self.netuid = netuid
        self.interval_seconds = interval_seconds

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the weight setter background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the weight setter thread and wait for it to finish.

        Uses a generous timeout to account for:
        - HTTP request to Backend (up to 30s)
        - Blockchain transaction with wait_for_inclusion (up to 60s)
        - Buffer time (10s)
        """
        self._stop_event.set()
        if self._thread is not None:
            join_timeout = 100  # 30s HTTP + 60s blockchain + 10s buffer
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logging.warning(
                    f"Weight setter thread did not stop within {join_timeout}s"
                )

    def _build_weights(
        self, top_miner_hotkey: str, emission_wt: float = 1.0
    ) -> list[float]:
        """Build weight vector with top miner and optional burn via UID 0.

        If the top miner is not in the metagraph (e.g. deregistered
        between leaderboard fetch and weight set), all emissions go to
        the UID 0 burn so weights are still set.
        """
        n = len(self.metagraph.hotkeys)
        if n == 0:
            return []

        weights = [0.0] * n
        try:
            top_idx = self.metagraph.hotkeys.index(top_miner_hotkey)
            weights[top_idx] = emission_wt
        except ValueError:
            logging.warning(
                f"Top miner {top_miner_hotkey} not found in metagraph, "
                "burning all emissions to UID 0"
            )
            emission_wt = 0.0
        weights[0] += 1.0 - emission_wt
        return weights

    def _run(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                self.metagraph.sync()
                top = self.backend_client.get_top_miner()
                logging.info(
                    f"Top miner: {top.top_miner_hotkey} with score {top.top_score}"
                )
                emission_wt = (
                    top.emission_weight
                    if isinstance(top.emission_weight, (int, float))
                    else 1.0
                )
                logging.info(
                    f"Emission weight: {emission_wt:.3f} (burn: {1.0 - emission_wt:.3f})"
                )
                weights = self._build_weights(
                    top.top_miner_hotkey, emission_wt=emission_wt
                )
                if not weights:
                    logging.warning("Skipping weight update (empty metagraph)")
                else:
                    self.subtensor.set_weights(
                        netuid=self.netuid,
                        wallet=self.wallet,
                        uids=self.metagraph.uids,
                        weights=weights,
                        wait_for_inclusion=True,
                    )
                    logging.info("Successfully set weights")
            except BackendError as e:
                if e.is_auth_error:
                    # Authentication errors are permanent - log and continue
                    # (will retry on next interval in case credentials are refreshed)
                    logging.error(f"Weight setting auth error (will retry): {e}")
                elif e.is_transient:
                    # Transient errors - log at warning level
                    logging.warning(f"Weight setting transient error: {e}")
                else:
                    # Other backend errors
                    logging.error(f"Weight setting backend error: {e}")
            except Exception as e:
                # Non-backend errors (e.g., blockchain issues)
                logging.error(f"Weight setting failed: {type(e).__name__}: {e}")

            self._stop_event.wait(self.interval_seconds)
