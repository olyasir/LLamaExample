/*!
 *  Copyright (c) 2023 by Contributors
 * \file serve/engine.h
 * \brief The header of serving engine in MLC LLM.
 */
#ifndef MLC_ENGINE_WITH_LOAD_H_
#define MLC_ENGINE_WITH_LOAD_H_

#include <tvm/runtime/packed_func.h>

#include <cpp/serve/data.h>
#include <cpp/serve/event_trace_recorder.h>
#include <cpp/serve/request.h>
#include <cpp/serve/request_state.h>

namespace mlc {
namespace llm {
namespace serve {

using namespace tvm::runtime;

typedef TypedPackedFunc<void(Array<RequestStreamOutput>)> FRequestStreamCallback;


class EngineWithLoad {
 public:
  /********************** Engine Management **********************/
  virtual ~EngineWithLoad() = default;

  virtual void LoadParams() =0;

  /*!
   * \brief Create an engine in unique pointer.
   * \param engine_config_json_str The serialized JSON string of the engine config.
   * \param device The device where the run models.
   * \param request_stream_callback The request stream callback function to.
   * \param trace_recorder Event trace recorder for requests.
   * \return The created Engine in pointer, and the default generation config.
   */
  static Result<EngineCreationOutput> Create(const std::string& engine_config_json_str,
                                             Device device,
                                             FRequestStreamCallback request_stream_callback,
                                             Optional<EventTraceRecorder> trace_recorder);

  /*! \brief Reset the engine, clean up all running data and metrics. */
  virtual void Reset() = 0;

  /*! \brief Check if the engine has no request to process. */
  virtual bool Empty() = 0;

  /*! \brief Get the request stream callback function of the engine. */
  virtual FRequestStreamCallback GetRequestStreamCallback() = 0;

  /*! \brief Set the request stream callback function of the engine. */
  virtual void SetRequestStreamCallback(FRequestStreamCallback request_stream_callback) = 0;

  /***************** High-level Request Management *****************/

  /*! \brief Add a new request to the engine. */
  virtual void AddRequest(Request request) = 0;

  /*! \brief Abort the input request (specified by id string) from engine. */
  virtual void AbortRequest(const String& request_id) = 0;

  /*! \brief Abort all requests from the engine. */
  virtual void AbortAllRequests() = 0;

  /*********************** Engine Action ***********************/

  /*!
   * \brief The main function that the engine takes a step of action.
   * At each step, the engine may decide to
   * - run prefill for one (or more) requests,
   * - run one-step decode for the all existing requests
   * ...
   * In the end of certain actions (e.g., decode), the engine will
   * check if any request has finished, and will return the
   * generation results for those finished requests.
   */
  virtual void Step() = 0;

  /************** Debug/Profile **************/

  /*! \brief Internal engine metrics. */
  virtual String JSONMetrics() = 0;

  /*! \brief Call the given global function on all workers. Only for debug purpose. */
  virtual void DebugCallFuncOnAllAllWorker(const String& func_name) = 0;
};

}  // namespace serve
}  // namespace llm
}  // namespace mlc

#endif  // MLC_ENGINE_WITH_LOAD_H_
