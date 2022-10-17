# -*- coding: utf-8 -*-
"""
HybridResult class
"""

from qat.comm.shared.ttypes import PostProcessedResult, ParsedPostProcessedResult


class HybridResult(PostProcessedResult, ParsedPostProcessedResult):
    """
    Result which is, at the same time, the output of Plugin.post_process
    but also BatchGenerator.post_process.

    Result is not parsed.

    Args:
        batch_result (:class:`~qat.core.BatchResult`): Wrapped result
    """

    def __init__(self, batch_result):
        PostProcessedResult.__init__(self, results=batch_result)
        ParsedPostProcessedResult.__init__(self, batch_result=batch_result)
