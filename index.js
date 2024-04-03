import OpenAI from "openai"
import 'dotenv/config'

const openAI = new OpenAI()
const results = await openAI.chat.completions.create({
  model: 'gpt-3.5-turbo',
  messages: [
    {
      role: 'system', 
      content: ''
    },
    {
      role: 'user',
      content: '',
    }
  ]
})

console.log(results.choices[0].message.content)
