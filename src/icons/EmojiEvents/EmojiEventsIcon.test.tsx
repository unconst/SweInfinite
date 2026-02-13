import { EmojiEventsIcon } from '../index';

describe('EmojiEventsIcon', () => {
  it('should be exported from icons module', () => {
    expect(EmojiEventsIcon).toBeDefined();
    expect(typeof EmojiEventsIcon).toBe('function');
  });

  it('should render with default props', () => {
    const result = EmojiEventsIcon({});
    expect(result).toBeDefined();
    expect(result.type).toBe('svg');
    expect(result.props.width).toBeDefined();
    expect(result.props.height).toBeDefined();
  });

  it('should accept custom width and height props', () => {
    const result = EmojiEventsIcon({ width: 32, height: 32 });
    expect(result.props.width).toBe(32);
    expect(result.props.height).toBe(32);
  });
});
