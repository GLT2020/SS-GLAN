pragma solidity ^0.5.2;
interface IERC20 {
    function transfer(address to, uint256 value) external returns (bool);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
    function totalSupply() external view returns (uint256);
    function balanceOf(address who) external view returns (uint256);
    function allowance(address owner, address spender) external view returns (uint256);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
pragma solidity ^0.5.2;
library SafeMath {
    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        if (a == 0) {
            return 0;
        }
        uint256 c = a * b;
        require(c / a == b);
        return c;
    }
    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b > 0);
        uint256 c = a / b;
        return c;
    }
    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b <= a);
        uint256 c = a - b;
        return c;
    }
    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        require(c >= a);
        return c;
    }
    function mod(uint256 a, uint256 b) internal pure returns (uint256) {
        require(b != 0);
        return a % b;
    }
}
pragma solidity ^0.5.2;
contract ERC20 is IERC20 {
    using SafeMath for uint256;
  address payable winner_TOD1;
function play_TOD1(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD1 = msg.sender;
        }
    }
function getReward_TOD1() payable public{
       winner_TOD1.transfer(msg.value);
    }
  mapping (address => uint256) private _balances;
  bool claimed_TOD2 = false;
address payable owner_TOD2;
uint256 reward_TOD2;
function setReward_TOD2() public payable {
        require (!claimed_TOD2);
        require(msg.sender == owner_TOD2);
        owner_TOD2.transfer(reward_TOD2);
        reward_TOD2 = msg.value;
    }
    function claimReward_TOD2(uint256 submission) public {
        require (!claimed_TOD2);
        require(submission < 10);
        msg.sender.transfer(reward_TOD2);
        claimed_TOD2 = true;
    }
  mapping (address => mapping (address => uint256)) private _allowed;
  address payable winner_TOD17;
function play_TOD17(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD17 = msg.sender;
        }
    }
function getReward_TOD17() payable public{
       winner_TOD17.transfer(msg.value);
    }
  uint256 private _totalSupply;
    function totalSupply() public view returns (uint256) {
        return _totalSupply;
    }
address payable winner_TOD25;
function play_TOD25(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD25 = msg.sender;
        }
    }
function getReward_TOD25() payable public{
       winner_TOD25.transfer(msg.value);
    }
    function balanceOf(address owner) public view returns (uint256) {
        return _balances[owner];
    }
address payable winner_TOD19;
function play_TOD19(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD19 = msg.sender;
        }
    }
function getReward_TOD19() payable public{
       winner_TOD19.transfer(msg.value);
    }
    function allowance(address owner, address spender) public view returns (uint256) {
        return _allowed[owner][spender];
    }
bool claimed_TOD26 = false;
address payable owner_TOD26;
uint256 reward_TOD26;
function setReward_TOD26() public payable {
        require (!claimed_TOD26);
        require(msg.sender == owner_TOD26);
        owner_TOD26.transfer(reward_TOD26);
        reward_TOD26 = msg.value;
    }
    function claimReward_TOD26(uint256 submission) public {
        require (!claimed_TOD26);
        require(submission < 10);
        msg.sender.transfer(reward_TOD26);
        claimed_TOD26 = true;
    }
    function transfer(address to, uint256 value) public returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }
bool claimed_TOD20 = false;
address payable owner_TOD20;
uint256 reward_TOD20;
function setReward_TOD20() public payable {
        require (!claimed_TOD20);
        require(msg.sender == owner_TOD20);
        owner_TOD20.transfer(reward_TOD20);
        reward_TOD20 = msg.value;
    }
    function claimReward_TOD20(uint256 submission) public {
        require (!claimed_TOD20);
        require(submission < 10);
        msg.sender.transfer(reward_TOD20);
        claimed_TOD20 = true;
    }
    function approve(address spender, uint256 value) public returns (bool) {
        _approve(msg.sender, spender, value);
        return true;
    }
bool claimed_TOD32 = false;
address payable owner_TOD32;
uint256 reward_TOD32;
function setReward_TOD32() public payable {
        require (!claimed_TOD32);
        require(msg.sender == owner_TOD32);
        owner_TOD32.transfer(reward_TOD32);
        reward_TOD32 = msg.value;
    }
    function claimReward_TOD32(uint256 submission) public {
        require (!claimed_TOD32);
        require(submission < 10);
        msg.sender.transfer(reward_TOD32);
        claimed_TOD32 = true;
    }
    function transferFrom(address from, address to, uint256 value) public returns (bool) {
        _transfer(from, to, value);
        _approve(from, msg.sender, _allowed[from][msg.sender].sub(value));
        return true;
    }
bool claimed_TOD38 = false;
address payable owner_TOD38;
uint256 reward_TOD38;
function setReward_TOD38() public payable {
        require (!claimed_TOD38);
        require(msg.sender == owner_TOD38);
        owner_TOD38.transfer(reward_TOD38);
        reward_TOD38 = msg.value;
    }
    function claimReward_TOD38(uint256 submission) public {
        require (!claimed_TOD38);
        require(submission < 10);
        msg.sender.transfer(reward_TOD38);
        claimed_TOD38 = true;
    }
    function increaseAllowance(address spender, uint256 addedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowed[msg.sender][spender].add(addedValue));
        return true;
    }
bool claimed_TOD4 = false;
address payable owner_TOD4;
uint256 reward_TOD4;
function setReward_TOD4() public payable {
        require (!claimed_TOD4);
        require(msg.sender == owner_TOD4);
        owner_TOD4.transfer(reward_TOD4);
        reward_TOD4 = msg.value;
    }
    function claimReward_TOD4(uint256 submission) public {
        require (!claimed_TOD4);
        require(submission < 10);
        msg.sender.transfer(reward_TOD4);
        claimed_TOD4 = true;
    }
    function decreaseAllowance(address spender, uint256 subtractedValue) public returns (bool) {
        _approve(msg.sender, spender, _allowed[msg.sender][spender].sub(subtractedValue));
        return true;
    }
address payable winner_TOD7;
function play_TOD7(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD7 = msg.sender;
        }
    }
function getReward_TOD7() payable public{
       winner_TOD7.transfer(msg.value);
    }
    function _transfer(address from, address to, uint256 value) internal {
        require(to != address(0));
        _balances[from] = _balances[from].sub(value);
        _balances[to] = _balances[to].add(value);
        emit Transfer(from, to, value);
    }
address payable winner_TOD23;
function play_TOD23(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD23 = msg.sender;
        }
    }
function getReward_TOD23() payable public{
       winner_TOD23.transfer(msg.value);
    }
    function _mint(address account, uint256 value) internal {
        require(account != address(0));
        _totalSupply = _totalSupply.add(value);
        _balances[account] = _balances[account].add(value);
        emit Transfer(address(0), account, value);
    }
bool claimed_TOD14 = false;
address payable owner_TOD14;
uint256 reward_TOD14;
function setReward_TOD14() public payable {
        require (!claimed_TOD14);
        require(msg.sender == owner_TOD14);
        owner_TOD14.transfer(reward_TOD14);
        reward_TOD14 = msg.value;
    }
    function claimReward_TOD14(uint256 submission) public {
        require (!claimed_TOD14);
        require(submission < 10);
        msg.sender.transfer(reward_TOD14);
        claimed_TOD14 = true;
    }
    function _burn(address account, uint256 value) internal {
        require(account != address(0));
        _totalSupply = _totalSupply.sub(value);
        _balances[account] = _balances[account].sub(value);
        emit Transfer(account, address(0), value);
    }
bool claimed_TOD30 = false;
address payable owner_TOD30;
uint256 reward_TOD30;
function setReward_TOD30() public payable {
        require (!claimed_TOD30);
        require(msg.sender == owner_TOD30);
        owner_TOD30.transfer(reward_TOD30);
        reward_TOD30 = msg.value;
    }
    function claimReward_TOD30(uint256 submission) public {
        require (!claimed_TOD30);
        require(submission < 10);
        msg.sender.transfer(reward_TOD30);
        claimed_TOD30 = true;
    }
    function _approve(address owner, address spender, uint256 value) internal {
        require(spender != address(0));
        require(owner != address(0));
        _allowed[owner][spender] = value;
        emit Approval(owner, spender, value);
    }
bool claimed_TOD8 = false;
address payable owner_TOD8;
uint256 reward_TOD8;
function setReward_TOD8() public payable {
        require (!claimed_TOD8);
        require(msg.sender == owner_TOD8);
        owner_TOD8.transfer(reward_TOD8);
        reward_TOD8 = msg.value;
    }
    function claimReward_TOD8(uint256 submission) public {
        require (!claimed_TOD8);
        require(submission < 10);
        msg.sender.transfer(reward_TOD8);
        claimed_TOD8 = true;
    }
    function _burnFrom(address account, uint256 value) internal {
        _burn(account, value);
        _approve(account, msg.sender, _allowed[account][msg.sender].sub(value));
    }
address payable winner_TOD39;
function play_TOD39(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD39 = msg.sender;
        }
    }
function getReward_TOD39() payable public{
       winner_TOD39.transfer(msg.value);
    }
}
pragma solidity ^0.5.2;
contract ERC20Burnable is ERC20 {
    function burn(uint256 value) public {
        _burn(msg.sender, value);
    }
bool claimed_TOD36 = false;
address payable owner_TOD36;
uint256 reward_TOD36;
function setReward_TOD36() public payable {
        require (!claimed_TOD36);
        require(msg.sender == owner_TOD36);
        owner_TOD36.transfer(reward_TOD36);
        reward_TOD36 = msg.value;
    }
    function claimReward_TOD36(uint256 submission) public {
        require (!claimed_TOD36);
        require(submission < 10);
        msg.sender.transfer(reward_TOD36);
        claimed_TOD36 = true;
    }
    function burnFrom(address from, uint256 value) public {
        _burnFrom(from, value);
    }
address payable winner_TOD35;
function play_TOD35(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD35 = msg.sender;
        }
    }
function getReward_TOD35() payable public{
       winner_TOD35.transfer(msg.value);
    }
}
pragma solidity ^0.5.2;
contract ERC20Detailed is IERC20 {
  address payable winner_TOD37;
function play_TOD37(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD37 = msg.sender;
        }
    }
function getReward_TOD37() payable public{
       winner_TOD37.transfer(msg.value);
    }
  string private _name;
  address payable winner_TOD3;
function play_TOD3(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD3 = msg.sender;
        }
    }
function getReward_TOD3() payable public{
       winner_TOD3.transfer(msg.value);
    }
  string private _symbol;
  address payable winner_TOD9;
function play_TOD9(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD9 = msg.sender;
        }
    }
function getReward_TOD9() payable public{
       winner_TOD9.transfer(msg.value);
    }
  uint8 private _decimals;
    constructor (string memory name, string memory symbol, uint8 decimals) public {
        _name = name;
        _symbol = symbol;
        _decimals = decimals;
    }
bool claimed_TOD40 = false;
address payable owner_TOD40;
uint256 reward_TOD40;
function setReward_TOD40() public payable {
        require (!claimed_TOD40);
        require(msg.sender == owner_TOD40);
        owner_TOD40.transfer(reward_TOD40);
        reward_TOD40 = msg.value;
    }
    function claimReward_TOD40(uint256 submission) public {
        require (!claimed_TOD40);
        require(submission < 10);
        msg.sender.transfer(reward_TOD40);
        claimed_TOD40 = true;
    }
    function name() public view returns (string memory) {
        return _name;
    }
address payable winner_TOD33;
function play_TOD33(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD33 = msg.sender;
        }
    }
function getReward_TOD33() payable public{
       winner_TOD33.transfer(msg.value);
    }
    function symbol() public view returns (string memory) {
        return _symbol;
    }
address payable winner_TOD27;
function play_TOD27(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD27 = msg.sender;
        }
    }
function getReward_TOD27() payable public{
       winner_TOD27.transfer(msg.value);
    }
    function decimals() public view returns (uint8) {
        return _decimals;
    }
address payable winner_TOD31;
function play_TOD31(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD31 = msg.sender;
        }
    }
function getReward_TOD31() payable public{
       winner_TOD31.transfer(msg.value);
    }
}
pragma solidity ^0.5.0;
contract AGR is ERC20, ERC20Detailed, ERC20Burnable {
  constructor() ERC20Detailed('Aggregion Token', 'AGR', 4) public {
        super._mint(msg.sender, 30000000000000);
  }
address payable winner_TOD13;
function play_TOD13(bytes32 guess) public{
       if (keccak256(abi.encode(guess)) == keccak256(abi.encode('hello'))) {
            winner_TOD13 = msg.sender;
        }
    }
function getReward_TOD13() payable public{
       winner_TOD13.transfer(msg.value);
    }
}