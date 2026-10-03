#-----------------------------------------------------------------------------------------------------------------------
# Module:  progress.py
# Project: classifyimages
# Version: 1.0.0
# Date:    2023-03-01
# Author:  Rohin Gosling
#
# Description:
#
#   Render counted pipeline progress in place on a terminal and compact final lines when output is redirected.
#-----------------------------------------------------------------------------------------------------------------------

from time   import perf_counter
from typing import TextIO


#-----------------------------------------------------------------------------------------------------------------------
# Class: ProgressDisplay
#
# Description:
#
#   Show completed-work percentages without filling redirected logs with transient updates.
#
# Attributes:
#
#   output      : Text stream receiving progress lines.
#   width       : Minimum character width for stage labels.
#   interactive : Whether the stream supports in-place terminal updates.
#   label       : Active stage label, or None between stages.
#   percentage  : Most recently rendered percentage.
#   started_at  : Stage start time from the monotonic performance clock, or None between stages.
#-----------------------------------------------------------------------------------------------------------------------

class ProgressDisplay:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Initialize the output stream, display width, terminal mode, and inactive stage state.
    #
    # Arguments:
    #
    #   output : Text stream receiving settings, progress, or summary output.
    #   width  : Minimum character width for stage labels.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, output: TextIO, width: int = 33 ) -> None:

        # Initialize the output stream, display width, terminal mode, and inactive stage state.

        self.output      = output
        self.width       = width
        self.interactive = bool ( getattr ( output, "isatty", lambda: False ) () )
        self.label       = None
        self.percentage  = -1
        self.started_at  = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: _render
    #
    # Description:
    #
    #   Render a changed percentage with elapsed time, using carriage returns for live terminal updates.
    #
    # Arguments:
    #
    #   percentage : Completed percentage to display.
    #   final      : Whether this update completes the stage and terminates its line.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def _render ( self, percentage: int, *, final: bool = False ) -> None:

        # Render a changed percentage with elapsed time, using carriage returns for live terminal updates.

        if self.label is None:

            # Return to caller because no stage is active.

            return
        if not final and percentage == self.percentage:

            # Return to caller because the displayed percentage has not changed.

            return
        self.percentage       = percentage
        elapsed_milliseconds  = round ( max ( 0.0, perf_counter () - self.started_at ) * 1000 )
        minutes, remainder    = divmod ( elapsed_milliseconds, 60_000 )
        seconds, milliseconds = divmod ( remainder, 1000 )
        line                  = f"{self.label:<{self.width}} {percentage:3d} % ({minutes:02d}:{seconds:02d}.{milliseconds:03d})"
        self.output.write ( f"\r{line}" if self.interactive and not final else line + "\n" )
        self.output.flush ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: begin
    #
    # Description:
    #
    #   Start one named progress stage and show its initial state on interactive terminals.
    #
    # Arguments:
    #
    #   label : Setting or progress-stage label to display.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def begin ( self, label: str ) -> None:

        # Start one named progress stage and show its initial state on interactive terminals.

        if self.label is not None:
            raise RuntimeError ( "A progress stage is already active" )
        self.label      = label
        self.percentage = -1
        self.started_at = perf_counter ()
        if self.interactive:
            self._render ( 0 )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: update
    #
    # Description:
    #
    #   Advance an active interactive stage while reserving 100 percent for completion.
    #
    # Arguments:
    #
    #   completed : Number of completed work units in the current step.
    #   total     : Total work units in the current step.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def update ( self, completed: int, total: int ) -> None:

        # Advance an active interactive stage while reserving 100 percent for completion.

        if self.label is None:
            raise RuntimeError ( "No progress stage is active" )
        if total > 0 and self.interactive:
            self._render ( min ( 99, completed * 100 // total ) )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: complete
    #
    # Description:
    #
    #   Finish the active stage at 100 percent and clear its timing state.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def complete ( self ) -> None:

        # Finish the active stage at 100 percent and clear its timing state.

        if self.label is None:
            raise RuntimeError ( "No progress stage is active" )
        if self.interactive:
            self.output.write ( "\r" )
        self._render ( 100, final = True )
        self.label      = None
        self.started_at = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: skip
    #
    # Description:
    #
    #   Print a reason for a skipped stage without starting a progress timer.
    #
    # Arguments:
    #
    #   label  : Setting or progress-stage label to display.
    #   reason : Explanation printed in place of a progress percentage.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def skip ( self, label: str, reason: str ) -> None:

        # Print a reason for a skipped stage without starting a progress timer.

        if self.label is not None:
            raise RuntimeError ( "A progress stage is already active" )
        self.output.write ( f"{label:<{self.width}} {reason}\n" )
        self.output.flush ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: abort
    #
    # Description:
    #
    #   End an interactive progress line and discard the active stage state.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #-------------------------------------------------------------------------------------------------------------------

    def abort ( self ) -> None:

        # End an interactive progress line and discard the active stage state.

        if self.label is not None and self.interactive:
            self.output.write ( "\n" )
            self.output.flush ()
        self.label      = None
        self.started_at = None
