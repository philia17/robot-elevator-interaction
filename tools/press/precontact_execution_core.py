"""Compatibility re-export for the installed pre-contact execution core.

The canonical implementation lives in the ``roarm_precontact_adapter`` Python
package.  Keep this module logic-free so offline tools cannot drift from the
runtime adapter's admission contract.
"""

from roarm_precontact_adapter.precontact_execution_core import *  # noqa: F401,F403
from roarm_precontact_adapter.precontact_execution_core import __all__
