import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    // Parse the request body
    const body = await request.json();
    
    // Validate required fields
    if (!body.abstract || typeof body.abstract !== 'string') {
      return NextResponse.json(
        { error: 'Abstract is required and must be a string' },
        { status: 400 }
      );
    }

    // Extract data from request
    const { title, abstract, authors, year, venue } = body;

    // Generate summary using AI service
    // You can replace this with your preferred AI service (OpenAI, Anthropic, etc.)
    const summary = await generateSummary(abstract, title);

    // Return the response as JSON
    return NextResponse.json({
      success: true,
      data: {
        title: title || 'not specified',
        abstract: abstract,
        summary: summary,
        authors: authors || [],
        year: year || null,
        venue: venue || null,
      },
    });
  } catch (error) {
    console.error('Error generating summary:', error);
    
    // Return error response
    return NextResponse.json(
      { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to generate summary' 
      },
      { status: 500 }
    );
  }
}

async function generateSummary(abstract: string, title?: string): Promise<string> {
  // Check if OpenAI API key is available
  const apiKey = process.env.OPENAI_API_KEY;
  
  if (!apiKey) {
    // Fallback: Return a basic summary if no API key is configured
    return `This is a summary of the publication "${title || 'Untitled'}". ${abstract.substring(0, 200)}...`;
  }

  try {
    // Call OpenAI API to generate summary
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-3.5-turbo',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant that generates concise summaries of academic publications. Your summaries should be clear, informative, and highlight the key contributions of the research.',
          },
          {
            role: 'user',
            content: `Please generate a concise summary of the following academic publication:\n\nTitle: ${title || 'Not specified'}\n\nAbstract: ${abstract}`,
          },
        ],
        max_tokens: 300,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(`OpenAI API error: ${response.status} ${JSON.stringify(errorData)}`);
    }

    const data = await response.json();
    const summary = data.choices?.[0]?.message?.content;

    if (!summary) {
      throw new Error('No summary generated from OpenAI API');
    }

    return summary;
  } catch (error) {
    console.error('Error calling OpenAI API:', error);
    // Fallback to basic summary if API call fails
    return `Summary of "${title || 'Untitled'}": ${abstract.substring(0, 200)}...`;
  }
}

// Handle OPTIONS request for CORS (if needed)
export async function OPTIONS() {
  return NextResponse.json({}, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}

