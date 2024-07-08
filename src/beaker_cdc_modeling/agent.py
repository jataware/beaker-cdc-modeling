from beaker_bunsen.bunsen_agent import BunsenAgent
from archytas.tool_utils import AgentRef, LoopControllerRef, tool

class CDCAgent(BunsenAgent):
    """
    You are a scientific assistant helping with epidemiological modeling and analytics.
    When reasonable, err on the side of calling the `generate_code` tool when the user asks you for help as you are
    working with them in a notebook environment and this will add the code in the notebook as a code cell. However, if
    the user asks you an informational question that does not call for code or is better answered via text, go ahead and
    use final_answer to communicate with the user directly.
    Of course, as always, don't hesitate to run tools to collect more information or do other tasks needed to complete
    the user's request.
    """

    @tool()
    async def search_cdc_api(self, query: str, agent: AgentRef) -> str:
        """
        This tool allows you to search the Centers for Disease Control and Prevention (CDC) 
        Data API (https://data.cdc.gov/) for datasets that match a user's query. 
        
        When you tell the user about the results, just give them the name of the dataset; 
        no need to provide information on data access/link.

        Args:
            query (str): A query to search over the CDC API.
        Returns:
            str: A list of datasets that match the user's query.
        """
        code = agent.context.get_code("search_cdc", {"query": query})
        response = await agent.context.evaluate(code)
        return response["return"]

    @tool()
    async def fetch_cdc_dataset(self, endpoint: str, agent: AgentRef) -> str:
        """
        This tool allows you to fetch a datasets from the Centers for Disease Control and Prevention (CDC)
        Data API based on a specifc dataset of interest. This should be used after the user has identified a 
        dataset with the `search_cdc_api` tool.

        Show the user the first few rows of the dataset as a Pandas DataFrame.

        Args:
            endpoint (str): a dataset endpoint of interest.
        Returns:
            str: the dataset as a Pandas DataFrame called `data` which can be used for further analysis
        """
        code = agent.context.get_code("fetch_cdc_dataset", {"endpoint": endpoint})
        response = await agent.context.evaluate(code)
        return response["return"]        