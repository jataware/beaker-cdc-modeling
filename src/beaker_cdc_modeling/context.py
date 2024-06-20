
from beaker_bunsen.bunsen_context import BunsenContext

from .agent import CDCAgent


class CDCContext(BunsenContext):

    agent_cls = CDCAgent
    enabled_subkernels = ["python3"]

    @classmethod
    def default_payload(cls) -> str:
        return "{}"

    # @property
    # def slug(self):
    #     return "cdc_cfa"
