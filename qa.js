import 'dotenv/config'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import { YoutubeLoader } from 'langchain/document_loaders/web/youtube'
import { CharacterTextSplitter } from 'langchain/text_splitter'
import { PDFLoader } from 'langchain/document_loaders/fs/pdf' 
import { openai } from './openai.js'

const question = process.argv[2] || 'hi'

const video1 = `https://youtu.be/EjfzPg1c1lY?si=zMtZEPMOyHftR1cq`
const video2 = 'https://youtu.be/9PNwYHqKnwo?si=JtIJ4ote9y5F5WKh'
const book1 = 'Hindi. Socio-Political and Economic Translation'
const book2 = 'Russian-English phrasebook'

export const createStore = (docs) =>
  MemoryVectorStore.fromDocuments(docs, new OpenAIEmbeddings())

export const docsFromYTVideo = async (video) => {
  const loader = YoutubeLoader.createFromUrl(video)
  return loader.loadAndSplit(
    new CharacterTextSplitter({
      separator: ' ',
      chunkSize: 2500,
      chunkOverlap: 100,
    })
  )
}

export const docsFromPDF = (pdfFilePath) => {
  const loader = new PDFLoader(pdfFilePath) 
  return loader.loadAndSplit(
    new CharacterTextSplitter({
      separator: '.',
      chunkSize: 2500,
      chunkOverlap: 200,
    })
  )
}

const loadStore = async () => {
  const videoDocs1 = await docsFromYTVideo(video1)
  const videoDocs2 = await docsFromYTVideo(video2)
  const pdfDocs1 = await docsFromPDF('yazyk-hindi-obshchestvenno-politicheskij-i-ekonomicheskij-perevod_RuLit_Me_856318.pdf') 
  const pdfDocs2 = await docsFromPDF('openai_learning/Russko-angliyskiy-razgovornik_RuLit_Me_624538.pdf') 

  return createStore([...videoDocs1, ...videoDocs2, ...pdfDocs1, ...pdfDocs2])
}

const query = async () => {
  const store = await loadStore()
  const results = await store.similaritySearch(question, 2)

  const response = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo-16k-0613',
    temperature: 0,
    messages: [
      {
        role: 'assistant',
        content:
          'Dear User,Please ask your question or specify your request so that I can help you. Be as precise and specific as possible in your query to get the most helpful response.Sincerely,Your Assistant'
      },
      {
        role: 'user',
        content: `I would like to ask a question or share a problem. Please pay attention to my comments and try to help me as best you can. I will be grateful for your support! Answer the following question using the provided context. If you cannot answer the question with the context, don't lie and make up stuff. Just say you need more context.
        Question: ${question}
  
        Context: ${results.map((r) => r.pageContent).join('\n')}`,
      },
    ],
  })
  console.log(
    `Answer: ${response.choices[0].message.content}`
  )
}

query()
