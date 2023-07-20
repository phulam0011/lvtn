import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Đưa ra cuộc trò chuyện sau đây và một câu hỏi tiếp theo, hãy diễn đạt lại câu hỏi tiếp theo thành một câu hỏi độc lập.

Lịch sử trò chuyện:
{chat_history}
Theo dõi đầu vào: {question}
Câu hỏi độc lập:`;

const QA_PROMPT = `Bạn là một trợ lý SmartShark hữu ích. Bạn được cung cấp các phần được trích xuất sau đây của một tài liệu dài và một câu hỏi. Cung cấp một câu trả lời đàm thoại dựa trên ngữ cảnh được cung cấp.
Bạn chỉ nên cung cấp các siêu liên kết tham chiếu ngữ cảnh bên dưới. KHÔNG tạo nên hyperlinks.
Nếu bạn không thể tìm thấy câu trả lời trong ngữ cảnh bên dưới, chỉ cần nói "Xin lỗi bạn, tôi không chắc là mình biết câu trả lời chính xác." Đừng cố tạo ra một câu trả lời.
Nếu câu hỏi không liên quan đến ngữ cảnh, hãy trả lời một cách lịch sự rằng bạn được điều chỉnh để chỉ trả lời những câu hỏi liên quan đến ngữ cảnh.
Phải cung cấp câu trả lời của bạn bằng ngôn ngữ tiếng Việt.
{context}

Câu hỏi: {question}
ANSWER_LANGUAGE = Vietnamese
Câu trả lời hữu ích trong markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo-16k', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: false, //The number of source documents returned is 4 by default
    
      // answerLanguage: 'Vietnamese',
    },
  );
  return chain;
};
