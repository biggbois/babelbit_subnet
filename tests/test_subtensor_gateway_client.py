import pytest
from unittest.mock import AsyncMock, Mock, patch

from babelbit.utils.subtensor_gateway_client import SubtensorGatewayClient


@pytest.mark.asyncio
async def test_post_json_reports_plain_text_gateway_errors():
    client = SubtensorGatewayClient(base_url="http://gw.test")
    response = AsyncMock()
    response.status = 500
    response.text = AsyncMock(return_value="500 Internal Server Error")

    session = Mock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=response)
    session.post.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "babelbit.utils.subtensor_gateway_client._get_session",
        new=AsyncMock(return_value=session),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            await client.get_current_block()

    assert (
        str(exc_info.value)
        == "gateway /v1/block/current failed status=500 body=500 Internal Server Error"
    )
